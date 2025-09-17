from datetime import datetime, timezone

# Boosts requested in the brief
PERSON_BOOST = 0.4
ROLE_BOOST = 0.2
PLACE_BOOST = 0.1

def score_prompt(prompt, tag_weights: dict, recent: list[str], targets: dict) -> float:
    """
    prompt.tags is a list of Tag objects (with .slug)
    tag_weights is a dict[tag_slug -> weight]
    recent is a list of e.g. "tag:<slug>" strings (recent history)
    targets is an optional dict[tag_slug -> weight] for nudging
    """
    T = {t.slug for t in getattr(prompt, "tags", [])}

    # Base matching against user's profile weights
    base = sum(tag_weights.get(t, 0.0) for t in T)

    # Preference nudges
    novelty = 0.15 * sum(1 for t in T if f"tag:{t}" not in recent)
    diversity = 0.20 * sum(1 for t in T if f"tag:{t}" in set(recent[:5]))
    pref = sum(targets.get(t, 0.0) for t in T)

    # Freshness (favor newer prompts a bit)
    p_dt = getattr(prompt, "created_at", None)
    age_days = (datetime.now(timezone.utc) - p_dt).days if p_dt else 999
    fresh = min(0.3, max(0.0, (30 - age_days) * 0.01))

    # --- New: contextual boosts ---
    has_person = any(t.startswith("person:") and t in tag_weights for t in T)
    has_role   = any(t.startswith("role:")   and t in tag_weights for t in T)
    has_place  = any(t.startswith("place:")  and t in tag_weights for t in T)

    boost = 0.0
    if has_person:
        boost += PERSON_BOOST
    if has_role:
        boost += ROLE_BOOST
    if has_place:
        boost += PLACE_BOOST

    return base + pref + novelty + fresh - diversity + boost

def morph_profile(profile: dict, used_tags: set[str]) -> dict:
    """
    Same as before; light reinforcement learning on tag weights.
    """
    w = profile.setdefault("tagWeights", {})
    for t in used_tags:
        w[t] = min(3.0, w.get(t, 0.0) + 0.15)
    for t in list(w.keys()):
        if t not in used_tags:
            w[t] = max(0.05, w[t] * 0.98)
    return profile
