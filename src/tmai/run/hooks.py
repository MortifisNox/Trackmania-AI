"""Hooks the trainer calls at fixed points of a run. Subclass TrainerHooks and
override what you need; combine several with HookList.

Call order, all from the trainer's driving (main) thread:
    on_start(trainer)      after the trainer is built, before the first episode
    on_step(tel, info)     after every control step (keep it to microseconds)
    should_reset_episode() every control step; True ends the attempt now
                           (end "reset") so the next one starts from the line
    on_episode(row)        after each episode's row is written to episodes.csv
    should_pause()         between episodes; True holds the driver until False
    on_stop(trainer)       once, after the driver and learner have stopped
"""

from __future__ import annotations


class TrainerHooks:
    def on_start(self, trainer) -> None:
        pass

    def on_step(self, tel, info) -> None:
        pass

    def on_episode(self, row: dict) -> None:
        pass

    def should_reset_episode(self) -> bool:
        return False

    def should_pause(self) -> bool:
        return False

    def on_stop(self, trainer) -> None:
        pass


class HookList(TrainerHooks):
    """Calls each hook in order. on_stop runs every hook even if one fails,
    so a broken dashboard can never block the final save."""

    def __init__(self, *hooks: TrainerHooks):
        self.hooks = [h for h in hooks if h is not None]

    def on_start(self, trainer) -> None:
        for h in self.hooks:
            h.on_start(trainer)

    def on_step(self, tel, info) -> None:
        for h in self.hooks:
            h.on_step(tel, info)

    def on_episode(self, row: dict) -> None:
        for h in self.hooks:
            h.on_episode(row)

    def should_reset_episode(self) -> bool:
        return any(h.should_reset_episode() for h in self.hooks)

    def should_pause(self) -> bool:
        return any([h.should_pause() for h in self.hooks])   # ask every hook: some poll for a stop here

    def on_stop(self, trainer) -> None:
        errors = []
        for h in self.hooks:
            try:
                h.on_stop(trainer)
            except Exception as exc:   # keep stopping the other hooks
                errors.append(exc)
        if errors:
            raise errors[0]
