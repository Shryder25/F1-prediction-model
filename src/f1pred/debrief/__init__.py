"""Automated post-race debrief: predicted vs actual, strategy calls, tyres, undercuts.

warehouse marts + saved models + f1pred.strategy
      │
      ▼
build.build_debrief()  ──►  Debrief (JSON)  ──►  render: Markdown + HTML
                                                     │
                                                     ▼
                                   site.build_site(): GitHub Pages
"""

from f1pred.debrief.build import build_debrief, latest_race_id
from f1pred.debrief.model import Debrief

__all__ = ["Debrief", "build_debrief", "latest_race_id"]
