"""Stage 4: replay or follow a race lap by lap and stream predictions to a HUD.

source (WarehouseReplay | OpenF1Source) -> LapFeatureBuilder -> saved models
    -> RaceUpdate -> FastAPI Server-Sent Events -> HUD in the browser
"""
