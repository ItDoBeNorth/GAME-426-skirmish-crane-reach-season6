"""Choose a reachable tile, then a target, with the trained PyTorch policy.

We encode the acting unit, visible units, and zones once. Movement candidates read that
context through attention. Each target candidate points directly at its enemy's token.
Both decisions use legal options, with the first option breaking equal-score ties.
"""

from pathlib import Path

import torch
from encoder import (
    USE_MEMORY_FEATURES,
    describe_move,
    describe_target,
    encode_entities,
    home_zones,
    move_options,
    target_entity_indices,
    target_options,
)
from enemy_memory import EnemyMemory
from imitation_model import load_policy
from sandbox.crane import me, paths, roster, tile, units, visible, zone

# S5 task 9 (hybrid rule part): remember enemies out of sight and show them to the model. The
# model must be trained with the same setting, so this follows the encoder's switch (off: the memory
# learner lost to the s5-features-b learner; scores in encoder.USE_MEMORY_FEATURES).
USE_MEMORY = USE_MEMORY_FEATURES
# S5 learner fixes (hybrid rules on top of the network's tile scores). Measured on the t10 learner
# against Banner: idle archers stood still on 26% of their activations (teacher 4%), and footmen
# stood on a zone under archer fire 11.7 times per game (teacher 1.7). Each rule only replaces a
# "stay" pick, with the network's best-scoring tile among the allowed ones.
# Seeds 600-619 wins/40 (mean) for the t10 model: plain = Banner 40 (97.9), teacher 23 (54.6),
# Banner learner 40 (94.2); both rules = 40 (98.7), 26 (63.6), 40 (96.4), 21/40 vs plain t10;
# unstick only = 40 (98.7), 24 (58.3), 40 (98.3); answer only = 40 (97.9), 25 (58.5), 40 (95.7).
# UNSTICK_IDLE_ARCHERS: an archer with no enemy in sight that also stayed last activation moves.
# Idle archers stood still 8% of the time instead of 26%.
UNSTICK_IDLE_ARCHERS = True
# FOOTMEN_ANSWER_ARCHERS: a footman with no enemy adjacent, shot at by a visible enemy archer it
# cannot reach, takes a tile closer to that archer. It rarely fires: footmen see 4 tiles and
# archers shoot from 6, so the shooter is usually out of sight. Kept on: "both" scored best.
FOOTMEN_ANSWER_ARCHERS = True
# S5 task 11, DAgger (tools/dagger_round.sh, model runs/t11.pt): one round of learner-driven
# corrections lost to t10 (head-to-head 18/40; teacher 45.0 vs 54.6). The teacher copies label
# idle archers "stay" twice as often when learners play (15% vs 7.5%), because the teacher keeps
# archers behind its footmen and the learner's footmen are more passive. Not installed.
ARCHER_RANGE = 6
# S5 task 10, safety veto: if the chosen tile would likely kill us before our next activation and
# some reachable tile would not, take the network's best-scoring survivable tile instead. The
# estimate is rough (straight-line reach, base damage, +2 for a cavalry charge, -1 arrow damage in
# forest), so it only vetoes clearly lethal tiles. Measured first: the t10 learner dies less per
# activation than the teacher (vs Banner: archers 1.4% vs 2.4%, footmen 3.7% vs 4.1%, cavalry 22.6%
# vs 23.5%), so there is no learner-specific gap to close.
# Off: seeds 600-619 wins/40 (mean) with the veto: Banner 40 (97.8), teacher 21 (52.3), Banner
# learner 39 (94.9), and 7/40 (18.5) head-to-head against the same learner without it. It cut cavalry
# deaths from ~23% to ~9% per activation, but, as with the teacher's CAVALRY_* switches, cavalry
# aggression is worth its cost.
SAFETY_VETO = False
# S5 task 12, model size (same t10 data): width/layers 32/1 (21.7k parameters, 105 KB, 54.0% tile
# agreement), 64/2 (113.7k, 474 KB, 53.4%, installed) and 128/3 (572.8k, 2.3 MB, 52.4%, best epoch 9:
# memorises fastest). Head-to-head against the installed 64/2 learner: 32/1 14/40 (35.7), 128/3
# 15/40 (38.5), so 64/2 stays.

# S6 rules (Season 6 adds wasteland: entering a waste tile costs 2 HP). The S5 model was trained
# without wasteland, so it does not know waste hurts. Baseline on S6 maps, seeds 600-619 wins/40
# (mean): Banner 40 (100), S5 teacher 15 (36.6; was 51/80 = 63.6 in S5), S5 learner 20 (50.0).
# AVOID_WASTE (F1): never pick a move whose path enters a waste tile; each tile keeps its shortest
# waste-free path. Tiles reachable only through waste are kept only if every option is one.
# F1 alone: Banner 40 (99.3), S5 teacher 26 (64.6), S5 learner 27 (68.0). The biggest S6 gain.
# Seeds 600-619 wins/40 (mean) for F1 + the other rules:
#   F1+F2:    Banner 40 (100), teacher 24 (60.2), S5 learner 26 (66.1)
#   F1+F3:    Banner 40 (100), teacher 25 (61.6), S5 learner 32 (79.7)
#   F1+F2+F3: Banner 40 (100), teacher 25 (61.6), S5 learner 32 (79.3); 20/40 (49.9) vs F1+F3
AVOID_WASTE = True
# UNSTICK_IDLE_CAVALRY (F2, playtest: "a cavalry wasn't moving because it couldn't see enemies"):
# cavalry with no enemy in sight that also stayed last activation, and is not standing on a zone,
# moves toward a zone to capture (idle_goal_zone), else to the network's best non-stay tile.
# Neutral in score (20/40 head-to-head with vs without it, on top of F1+F3): it rarely fires, since
# cavalry live ~3.9 activations and are usually fighting. Kept on: it fixes the playtest bug.
UNSTICK_IDLE_CAVALRY = True
# ARCHER_KITE (F3, playtest: "no clear shoot-and-retreat kiting"): an archer whose chosen tile is in
# reach of a visible enemy footman or cavalry (movement + attack range, straight line) takes the
# network's best tile that is out of every such reach but still within ARCHER_RANGE of an enemy.
# Vs the S5 learner 27 -> 32/40 on top of F1. Vs the S5 teacher, archers shoot more (0.33 -> 0.51
# arrows per activation; 10.0 -> 8.2 tiles from the nearest enemy).
ARCHER_KITE = True
# S6 stretch, retrain on wasteland maps (tools/feature_round.sh s6_waste: 21.9k teacher decisions,
# seeds 0-19 vs Banner + self-play, model runs/s6_waste.pt, all three rules on; 50.4% tile agreement).
# Seeds 600-619: Banner 40 (98.9), S5 teacher 22 (52.1), S5 learner 26 (64.1), and 15/40 (37.3)
# head-to-head against the installed S5 model + rules. Not installed: the S5 model stays.
# S6 round 2 (open issues). SHARE_ARCHER_SIGHTINGS (G1): melee took ~97 damage per game from enemy
# archers it cannot see (footman vision 4, archer range 6). Every unit that sees enemy archers sends
# them to our footmen ("R<round>|S a<idx>.q.r.hp", the teacher's format); FOOTMEN_ANSWER_ARCHERS
# also closes on archers reported this round or last, unless the footman stands on a zone.
# Seeds 600-619 wins/40 (mean): Banner 40 (100), S5 teacher 25 (61.2), S5 learner 32 (78.7).
# Head-to-head vs rivals/s6_final (F1-F3): 23/40 (57.5) on 600-619 + 43/80 (53.9) on 640-679 =
# 66/120 (55.1). Melee damage from archers barely moved (98.1 -> 97.4 per game), since the rule only
# replaces a footman's "stay". Off: on the held-out seeds 900-939 (80 games each) it scored 37/80
# (45.5) vs s6_final, S5 teacher 50/80 (61.8; s6_final 52/80, 64.4), S5 learner 64/80 (79.7; s6_final
# 60/80, 74.7). 103/200 head-to-head overall: no real gain.
SHARE_ARCHER_SIGHTINGS = False
# SECURE_KILLS (G2, "kill opponents"; cavalry trade poorly): if a reachable tile lets this unit kill
# a visible enemy this strike (strike_damage >= its hit points), take the network's best such tile
# and name that enemy. A unit on a zone only takes kill tiles on a zone; an archer (ARCHER_KITE)
# only tiles out of melee reach.
# Off: G2 alone: Banner 40 (100), S5 teacher 26 (63.3), S5 learner 31 (75.8), head-to-head vs
# s6_final 61/120 (50.4). G1+G2: teacher 27 (64.8), S5 learner 32 (77.9), vs s6_final 61/120 (51.1),
# below G1 alone (55.1). Cavalry trade unchanged (dealt 1.61 vs 1.57, taken 3.03 vs 2.96 per
# activation); footmen on a zone under fire 8.5 -> 7.5 per game.
SECURE_KILLS = False
TYPE_PRIORITY = {"archer": 0, "cavalry": 1, "footman": 2}  # G2: which killable enemy to name first


def option_tensor(rows):
    """Add a batch dimension: one decision, several rows, several features per row."""
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)


@torch.inference_mode()
def choose_action(model, observation, remembered=(), override=None):
    """S5: override(observation, moves, scores) may return another move index (hybrid rules)."""
    # Read the battlefield once. Both scorers use this same context.
    # S5 task 9: remembered enemies (enemy_memory) join the visible units as their own tokens.
    entities, enemy_indices = encode_entities(observation, remembered)
    # A mask marks real rows. During play every row is real; training also has padding.
    entity_mask = torch.ones((1, len(entities)), dtype=torch.bool)
    context = model.encode(option_tensor(entities), entity_mask)

    # S6 F1: with wasteland on, keep a waste-free path per tile and mark tiles only waste reaches.
    wasted = None
    if AVOID_WASTE and observation["observation"].get("parameters", {}).get("wasteland"):
        moves, wasted = waste_free_moves(observation)
    else:
        moves = move_options(observation)
    targets = target_options(observation)
    if not moves or not targets:
        raise ValueError("The observation must contain legal movement and target options")
    # Score destinations, keeping one legal path to each. argmax keeps the first tie.
    move_rows = option_tensor([describe_move(observation, end, path, remembered) for path, end in moves])
    move_mask = torch.ones((1, len(moves)), dtype=torch.bool)
    scores = model.score_moves(context, entity_mask, move_rows, move_mask)
    if wasted is not None and any(wasted) and not all(wasted):
        scores[0][torch.tensor(wasted)] = float("-inf")  # S6 F1: never walk into waste
    chosen = int(scores[0].argmax())
    forced_target = None  # S6 G2: an enemy unit_id the override wants named
    if override is not None:
        replaced = override(observation, moves, scores[0].tolist(), chosen)
        if isinstance(replaced, tuple):
            replaced, forced_target = replaced
        chosen = chosen if replaced is None else replaced
    path, end = moves[chosen]
    if forced_target is not None:
        slot = next((index + 1 for index, entry in enumerate(roster.enemies(observation)) if entry["unit_id"] == forced_target), None)
        if slot in targets:
            return {"path": path, "target": slot}

    # Describe targets from the tile we actually chose. The pointers connect target IDs
    # to the enemy tokens in context; -1 selects the model's learned no-target value.
    target_rows = option_tensor([describe_target(observation, end, target) for target in targets])
    pointers = torch.tensor([target_entity_indices(observation, targets, enemy_indices)])
    target_mask = torch.ones((1, len(targets)), dtype=torch.bool)
    scores = model.score_targets(context, target_rows, pointers, target_mask)
    return {"path": path, "target": targets[int(scores[0].argmax())]}


def enters_waste(observation, here, path_id):
    """S6 F1: does this path step onto a waste tile? (Ported from teacher._enters_waste.)"""
    position = here
    for direction in paths.decode(path_id):
        position = tile.neighbors(position)[direction]
        if tile.terrain_at(observation, position)["feature"] == "waste":
            return True
    return False


def waste_free_moves(observation):
    """S6 F1: like encoder.move_options (stay first, shortest path per tile), but each tile keeps its
    shortest path that avoids waste. Returns (moves, wasted): wasted[i] is True when every legal
    path to tile i enters waste."""
    here = me.position(observation)
    options = {}  # (q, r) -> (path ID, end tile, enters waste)
    for path_id, allowed in enumerate(observation["action_mask"]["path"]):
        if not allowed:
            continue
        end = tile.at_path_end(here, path_id)
        key = (end["q"], end["r"])
        hurts = bool(path_id) and enters_waste(observation, here, path_id)
        kept = options.get(key)
        if kept is None or (kept[2] and not hurts):
            options[key] = (path_id, end, hurts)
    values = list(options.values())
    return [(path_id, end) for path_id, end, _ in values], [hurts for _, _, hurts in values]


def idle_goal_zone(observation):
    """S6 F2: the zone centre an idle unit heads for, or None without zones.

    Our home zone is left to the holder. Of the middle zone and the enemy's home zone, take the
    nearer one with no visible ally standing on it; if both have one, the enemy's home zone.
    """
    _, enemy_home = home_zones(observation)
    middle = tile.at_center(observation)
    centers = [area["center"] for area in zone.zones(observation) if area["center"] == middle]
    if enemy_home is not None:
        centers.append(enemy_home)
    if not centers:
        return None
    own_side = me.side(observation)

    def held(center):
        area = zone.at(observation, center)
        return area is not None and any(
            ally["side"] == own_side and zone.at(observation, ally["position"]) == area
            for ally in visible.allies(observation)
        )

    free = [center for center in centers if not held(center)]
    if free:
        here = me.position(observation)
        return min(free, key=lambda center: tile.distance(here, center))
    return enemy_home if enemy_home is not None else centers[0]


def melee_reach(observation, end):
    """S6 F3: is `end` within straight-line reach (movement + attack range) of a visible enemy
    footman or cavalry? Terrain and blocking units are ignored, so this errs on the cautious side."""
    for enemy in visible.enemies(observation):
        if enemy["type"] == "archer":
            continue
        stats = units.STATS[enemy["type"]]
        if tile.distance(end, enemy["position"]) <= stats.movement_points + stats.attack_range:
            return True
    return False


def strike_damage(observation, attacker, start, end, target):
    """S6 G2: our strike's damage from `end` on a visible `target` (environment.md damage rules).

    Base damage; cavalry charge +2 when it ends >= 3 tiles from `start`; hill onto lower ground +1,
    lower ground onto a hill -1; forest cover -1 at range > 1; a footman beside a visible allied
    footman (shield wall) -1. Forest and the shield wall cancel the charge. Never below 1.
    """
    damage = units.STATS[attacker].damage
    charge = attacker == "cavalry" and tile.distance(start, end) >= 3
    ground_from = tile.terrain_at(observation, end)["terrain"]
    ground_to = tile.terrain_at(observation, target["position"])
    if ground_from == "hill" and ground_to["terrain"] != "hill":
        damage += 1
    elif ground_from != "hill" and ground_to["terrain"] == "hill":
        damage -= 1
    if ground_to["feature"] == "forest":
        charge = False
        if tile.distance(end, target["position"]) > 1:
            damage -= 1
    if target["type"] == "footman" and any(
        other["type"] == "footman"
        and other["unit_id"] != target["unit_id"]
        and tile.distance(other["position"], target["position"]) == 1
        for other in visible.enemies(observation)
    ):
        damage -= 1
        charge = False
    if charge:
        damage += 2
    return max(1, damage)


def incoming_estimate(observation, end):
    """S5 task 10: rough damage the visible enemies could deal to `end` before our next activation."""
    in_forest = tile.terrain_at(observation, end)["feature"] == "forest"
    total = 0
    for enemy in visible.enemies(observation):
        stats = units.STATS[enemy["type"]]
        gap = tile.distance(end, enemy["position"])
        if gap > stats.movement_points + stats.attack_range:
            continue
        damage = stats.damage
        if enemy["type"] == "cavalry" and gap >= 4:
            damage += 2  # room to walk 3+ tiles first: a charge
        if enemy["type"] == "archer" and in_forest:
            damage -= 1
        total += damage
    return total


class Agent:
    def __init__(self):
        # A model this small runs fastest on one CPU thread. The setting applies to the whole
        # process, so we set it before the first inference.
        torch.set_num_threads(1)
        self.model = load_policy(Path(__file__).with_name("imitation-policy.pt"))
        self.memory = EnemyMemory()  # S5 task 9; reset() starts a fresh one each match

    def reset(self, seed, observation):
        # This learner carries no private memory between activations. Its weights stay fixed.
        # S5 task 9 (hybrid): it now keeps one rule-based memory of enemies out of sight
        # (enemy_memory.py), the same tracker collection used. USE_MEMORY switches it off.
        self.memory = EnemyMemory()
        self.stayed = False  # S5: did our previous activation end where it started?
        # S6 G1: enemy archers reported by allies (unit_id -> {"position", "round"}), this
        # activation's outgoing report and its recipients (our footmen's players, not us).
        self.reported_archers = {}
        self.report = None
        own_id = me.unit_id(observation)
        self.player_of = {entry["player"]: entry["unit_id"] for entry in roster.allies(observation)}
        self.footman_players = [
            entry["player"] for entry in roster.allies(observation) if entry["type"] == "footman" and entry["unit_id"] != own_id
        ][:5]

    def act(self, observation):
        here = me.position(observation)
        remembered = ()
        if USE_MEMORY:
            self.memory.update(observation)
            remembered = self.memory.remembered(observation)
        action = choose_action(self.model, observation, remembered, override=self.override)
        self.stayed = tile.at_path_end(here, action["path"]) == here
        if SHARE_ARCHER_SIGHTINGS:
            # S6 G1: report the enemy archers we see now; chat() sends it after this order.
            archers = [enemy for enemy in visible.enemies(observation) if enemy["type"] == "archer"]
            entries = " ".join(
                f"a{enemy['unit_id'].split('_')[2]}.{enemy['position']['q']}.{enemy['position']['r']}.{enemy['hit_points']}"
                for enemy in archers
            )
            self.report = f"R{observation['observation']['round']}|S {entries}"[:200] if archers else None
            self.enemy_side = "blue" if me.side(observation) == "red" else "red"
        return action

    def chat(self, inbox):
        """S6 G1: read allies' archer sightings, then send ours to our footmen (direct messages only;
        a broadcast would also reach the enemy)."""
        if not SHARE_ARCHER_SIGHTINGS:
            return None
        for message in inbox:
            self.read_report(message)
        if self.report and getattr(self, "footman_players", None):
            return [{"to": player, "text": self.report} for player in self.footman_players]
        return None

    def read_report(self, message):
        """S6 G1: merge the archer entries of one "R<round>|..." report; anything malformed is ignored."""
        text = message.get("text")
        if message.get("from") not in getattr(self, "player_of", {}) or not isinstance(text, str) or not text.startswith("R"):
            return
        try:
            sections = text.split("|")
            sent_round = int(sections[0][1:])
            for section in sections[1:]:
                tag, _, body = section.partition(" ")
                if tag not in ("A", "S"):
                    continue
                for entry in body.split():
                    code, q, r, _ = entry.split(".")
                    if code[0] != "a":
                        continue
                    unit_id = f"{getattr(self, 'enemy_side', 'blue')}_archer_{int(code[1:])}"
                    known = self.reported_archers.get(unit_id)
                    if known is None or known["round"] <= sent_round:
                        self.reported_archers[unit_id] = {"position": {"q": int(q), "r": int(r)}, "round": sent_round}
        except (ValueError, IndexError):
            return

    def fresh_reported_archers(self, observation):
        """S6 G1: reported archers from this round or the last, minus those our own sight disproves."""
        here = me.position(observation)
        current = observation["observation"]["round"]
        seen = {enemy["unit_id"] for enemy in visible.enemies(observation)}
        vision = units.STATS[me.unit_type(observation)].vision
        found = []
        for unit_id, known in getattr(self, "reported_archers", {}).items():
            if known["round"] < current - 1 or unit_id in seen:
                continue
            if tile.distance(here, known["position"]) <= vision:
                continue  # we would see it there, and we do not: it moved
            found.append({"unit_id": unit_id, "type": "archer", "position": known["position"]})
        return found

    def override(self, observation, moves, scores, chosen):
        """Replace some picks (see the switches at the top). None = keep; (index, enemy unit_id) =
        take that tile and name that enemy (S6 G2)."""
        replaced = self.replace_picks(observation, moves, scores, chosen)
        if SECURE_KILLS:
            kill = self.kill_choice(observation, moves, scores, chosen if replaced is None else replaced)
            if kill is not None:
                return kill
        return replaced

    def kill_choice(self, observation, moves, scores, index):
        """S6 G2: (tile index, enemy unit_id) for a strike that kills a visible enemy, or None."""
        enemies = visible.enemies(observation)
        if not enemies:
            return None
        here = me.position(observation)
        kind = me.unit_type(observation)
        reach = units.STATS[kind].attack_range

        def victim(end):
            killable = [
                enemy
                for enemy in enemies
                if tile.distance(end, enemy["position"]) <= reach
                and strike_damage(observation, kind, here, end, enemy) >= enemy["hit_points"]
            ]
            return min(killable, key=lambda enemy: (TYPE_PRIORITY[enemy["type"]], enemy["hit_points"]), default=None)

        kept = victim(moves[index][1])
        if kept is not None:
            return index, kept["unit_id"]
        on_zone = zone.at(observation, here) is not None
        options = []
        for i, (_, end) in enumerate(moves):
            if scores[i] == float("-inf"):
                continue
            if on_zone and zone.at(observation, end) is None:
                continue
            if kind == "archer" and ARCHER_KITE and melee_reach(observation, end):
                continue
            enemy = victim(end)
            if enemy is not None:
                options.append((i, enemy["unit_id"]))
        return max(options, key=lambda option: scores[option[0]]) if options else None

    def replace_picks(self, observation, moves, scores, chosen):
        """S5 learner fixes (+ S6 F3 kiting): replace some picks. None = keep."""
        replaced = self.replace_stay(observation, moves, scores, chosen)
        if ARCHER_KITE and me.unit_type(observation) == "archer":
            index = chosen if replaced is None else replaced
            if melee_reach(observation, moves[index][1]):
                enemies = visible.enemies(observation)
                kite = [
                    i
                    for i, (_, end) in enumerate(moves)
                    if scores[i] > float("-inf")
                    and not melee_reach(observation, end)
                    and any(tile.distance(end, enemy["position"]) <= ARCHER_RANGE for enemy in enemies)
                ]
                if kite:
                    replaced = max(kite, key=lambda i: scores[i])
        if SAFETY_VETO:
            index = chosen if replaced is None else replaced
            hit_points = observation["observation"]["self"]["hit_points"]
            if incoming_estimate(observation, moves[index][1]) >= hit_points:
                safe = [i for i, (_, end) in enumerate(moves) if incoming_estimate(observation, end) < hit_points]
                if safe:
                    return max(safe, key=lambda i: scores[i])
        return replaced

    def replace_stay(self, observation, moves, scores, chosen):
        """S5 learner fixes 1-2: replace some "stay" picks. None = keep."""
        here = me.position(observation)
        if moves[chosen][1] != here:
            return None
        unit_type = me.unit_type(observation)
        enemies = visible.enemies(observation)

        def best(allowed):
            # S6: skip tiles F1 ruled out (score -inf).
            options = [index for index, (_, end) in enumerate(moves) if allowed(end) and scores[index] > float("-inf")]
            return max(options, key=lambda index: scores[index]) if options else None

        if UNSTICK_IDLE_ARCHERS and unit_type == "archer" and not enemies and getattr(self, "stayed", False):
            return best(lambda end: end != here)
        # S6 F2: idle cavalry heads for a zone to capture (a unit already on a zone keeps scoring).
        if UNSTICK_IDLE_CAVALRY and unit_type == "cavalry" and not enemies and getattr(self, "stayed", False):
            if zone.at(observation, here) is not None:
                return None
            goal = idle_goal_zone(observation)
            if goal is not None:
                gap = tile.distance(here, goal)
                closer = best(lambda end: tile.distance(end, goal) < gap)
                if closer is not None:
                    return closer
            return best(lambda end: end != here)
        if FOOTMEN_ANSWER_ARCHERS and unit_type == "footman":
            if any(tile.distance(here, enemy["position"]) <= 1 for enemy in enemies):
                return None
            shooters = [
                enemy
                for enemy in enemies
                if enemy["type"] == "archer" and tile.distance(here, enemy["position"]) <= ARCHER_RANGE
            ]
            if not shooters and SHARE_ARCHER_SIGHTINGS and zone.at(observation, here) is None:
                # S6 G1: archers we cannot see but allies reported (a footman on a zone keeps scoring).
                shooters = [
                    archer
                    for archer in self.fresh_reported_archers(observation)
                    if tile.distance(here, archer["position"]) <= ARCHER_RANGE
                ]
            if shooters:
                archer = min(shooters, key=lambda enemy: tile.distance(here, enemy["position"]))
                gap = tile.distance(here, archer["position"])
                return best(lambda end: tile.distance(end, archer["position"]) < gap)
        return None
