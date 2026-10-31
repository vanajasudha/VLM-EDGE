# Warehouse pilot preparation

Current count: 0 permitted warehouse images, 0 genuine human-verified labels. The approximately 40-image/120-pair target is pending source availability and annotation quality.

    data/warehouse_pilot/
      images/                 registered permitted image copies
      sources.csv             source/permission/session/scene/split ledger; currently header only
      annotations/            append-only events.jsonl after the first human save
      permissions/            actual human-supplied permission evidence
      exports/                new verified-only manifests
      quality/                inventory, quality and preservation evidence
      ANNOTATION_GUIDE.md
      README.md

No genuine warehouse sources are supplied. Existing test_frames/ and sample_videos/ are repeated stylized forest demo material; they were not imported. No images were downloaded or generated.

Read ANNOTATION_GUIDE.md first. sources.csv contains no invented records. The following is a command pattern for a human with actual rights and image knowledge. Replace every uppercase placeholder with genuine information; do not execute placeholders.

    venv/bin/python tools/annotate_warehouse.py register /ACTUAL/PERMITTED/IMAGE.jpg --image-id UNIQUE_ID --source-origin "ACTUAL OWNER/CAPTURE ORIGIN" --rights-basis "ACTUAL PERMISSION OR LICENCE" --permission-evidence "ACTUAL DOCUMENT PATH OR TICKET" --rights-reviewer REAL_ID --session-id ACTUAL_SESSION --scene-id ACTUAL_SCENE --split test --source-id ACTUAL_CAMERA_IF_KNOWN --tags forklift,occupied --permission-confirmed --warehouse-confirmed

source-id is optional. Unknown origin, permission or grouping must be clarified rather than fabricated. The confirmation flags are human attestations. Coverage tags come from human inspection: forklift, pallet_truck, worker, boxes, pallets, loading_dock, storage_aisle, walkway_clear, walkway_obstructed, empty, occupied, challenging_lighting, occlusion.

Start the separate local annotation form:

    venv/bin/python tools/annotate_warehouse.py serve --port 8766

Open http://127.0.0.1:8766 manually. Inspect the image, enter question/type/reference/notes and actual identities, and explicitly confirm verification only after review. The verification checkbox is unchecked by default. Ctrl+C stops the server. No server is left running by preparation tests.

Quality and export:

    venv/bin/python tools/annotate_warehouse.py quality
    venv/bin/python tools/annotate_warehouse.py export

The initial empty dataset is NOT_READY and export correctly refuses it. Successful export validates the exact existing manifest schema and prints a new file path. No evaluation approval is created.
