# 🛣️ Road Damage Dataset: Potholes, Cracks & Manholes

A curated image dataset for detecting **potholes**, **cracks**, and **manholes** on road surfaces — built for object detection research, model training, and evaluation. Annotations are provided in three formats: raw polygons, YOLO, and COCO JSON.

---

## 📦 Dataset Structure

```
├── annotations_coco.json         # COCO-format annotations
├── COCO-conversion-script.py     # Polygon → COCO converter
├── YOLO-conversion-script.py     # Polygon → YOLO converter
├── images/                       # All dataset images (.jpg)
├── labels/                       # Original polygon annotations (.txt)
├── labels-YOLO/                  # YOLO-format bounding boxes (.txt)
└── README.md
```

| Folder / File | Description |
|---|---|
| `images/` | All dataset images in JPG format |
| `labels/` | Original polygon-based annotations — one `.txt` file per image, one line per object |
| `labels-YOLO/` | Axis-aligned bounding boxes in YOLO format, ready for direct training |
| `annotations_coco.json` | Full dataset annotations in COCO format (bounding boxes in pixels) |
| `YOLO-conversion-script.py` | Converts polygon labels → YOLO format |
| `COCO-conversion-script.py` | Converts polygon labels → COCO JSON format |

---

## 🏷️ Classes

| Class ID | Name |
|:---:|---|
| 0 | Pothole |
| 1 | Crack |
| 2 | Manhole |

---

## 📐 Annotation Formats

### 1. Polygon Labels (`labels/`)
- 4-point quadrilaterals per object: `class_id x1 y1 x2 y2 x3 y3 x4 y4`
- Coordinates normalized to `[0, 1]`
- Points are ordered and convex

### 2. YOLO Labels (`labels-YOLO/`)
- Axis-aligned bounding boxes derived from polygons
- Format: `class_id x_center y_center width height`
- All values normalized to `[0, 1]`

### 3. COCO JSON (`annotations_coco.json`)
- Bounding boxes in `[x_min, y_min, width, height]`, in **pixels**
- Optional segmentation polygons included

> 💡 Use `YOLO-conversion-script.py` or `COCO-conversion-script.py` to regenerate YOLO/COCO labels directly from the original polygon annotations — ensuring full compatibility with YOLO, COCO-based frameworks, and other detection/segmentation tools.

---

## 📷 Data Acquisition

| Attribute | Details |
|---|---|
| **Devices** | GoPro & smartphone cameras, vehicle-mounted at ~1.2–1.5 m height, angled 10–15° to the road plane |
| **Original resolution** | 1920×1080 (GoPro), 1280×720 (smartphone) — downsampled to **640×360** for release |
| **Road types** | Urban, suburban, and rural roads across Region X and Region Y |
| **Conditions** | Captured Feb–Mar 2025; varying lighting (sunny/cloudy), mild precipitation |
| **Vehicle speed** | 20–50 km/h |
| **Exclusions** | Severely blurred, occluded, or nighttime images excluded |
| **Stabilization** | Horizontal FOV stabilization mode used throughout |

---

## 🚀 Usage

- **YOLO training** → use the `labels-YOLO/` folder directly with your training config
- **COCO-based tasks** → use `annotations_coco.json` for detection/evaluation pipelines
- **Advanced usage** → original polygons in `labels/` support oriented bounding boxes or segmentation tasks
- **Reproducibility** → both conversion scripts are included so annotations can be regenerated from scratch

---

## 📖 Citation

If you use this dataset in your work, please cite:

```
https://doi.org/10.5281/zenodo.17834373
```

---

## 📄 License

*(Add your dataset's license here, e.g. CC BY 4.0, MIT, etc.)*
