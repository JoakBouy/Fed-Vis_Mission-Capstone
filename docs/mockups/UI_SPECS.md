# Fed-Vis UI Design Specifications

## Doctor's Cockpit - Main Dashboard

### Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Fed-Vis    Patient #2847 - Brain MRI    🔔  👤  ⚙️             │
├─────┬───────────────────────────────────────────────────────┬───────────────┤
│     │                                                       │               │
│ 🏠  │                                                       │ RESULTS       │
│     │                                                       │               │
│ 👥  │           3D BRAIN VISUALIZATION                      │ Dice: 0.89    │
│     │                                                       │ IoU:  0.82    │
│ 🧊  │          [Tumor region in orange]                     │ Vol: 12.4cm³  │
│     │                                                       │               │
│ 📊  │                                                       │ ─────────────│
│     │                                                       │ CONFIDENCE    │
│ ⚙️  │                                                       │ ████████░░ 87% │
│     │                                                       │               │
├─────┼───────────────────────────────────────────────────────┼───────────────┤
│     │  Axial ▼  |  Sagittal  |  Coronal  |  3D            │ Export ↓      │
└─────┴───────────────────────────────────────────────────────┴───────────────┘
```

### Color Palette

| Element           | Color       | Hex Code  |
|-------------------|-------------|-----------|
| Background        | Dark Navy   | #0D1117   |
| Sidebar           | Darker      | #161B22   |
| Accent Primary    | Blue        | #58A6FF   |
| Accent Secondary  | Purple      | #A371F7   |
| Tumor Highlight   | Orange      | #F97316   |
| Success           | Green       | #3FB950   |
| Text Primary      | White       | #E6EDF3   |
| Text Muted        | Gray        | #8B949E   |

### Components

#### 1. Header Bar (height: 64px)
- Logo: "Fed-Vis" with brain icon
- Patient info: Name, ID, scan date
- Icons: Notifications, User profile, Settings

#### 2. Sidebar (width: 72px)
- Active indicator: Left border accent
- Icons only, tooltip on hover
- Pages: Dashboard, Patients, 3D Viewer, Reports, Settings

#### 3. Main Viewport (flex-grow)
- 3D brain render (Three.js)
- Segmentation overlay toggle
- Mouse controls: Rotate, Zoom, Pan
- Slice navigation slider

#### 4. Results Panel (width: 280px)
- Segmentation metrics
- Confidence gauge
- Attention heatmap preview
- Export options

#### 5. Bottom Toolbar (height: 48px)
- View tabs: Axial, Sagittal, Coronal, 3D
- Slice slider
- Play/Pause animation

---

## Figma Component Specs

### Typography
- Font: Inter
- Headers: 18px Semi-bold
- Body: 14px Regular
- Metrics: 24px Bold (numbers)

### Spacing
- Base unit: 8px
- Sidebar padding: 16px
- Card padding: 24px
- Component gap: 12px

### Border Radius
- Cards: 12px
- Buttons: 8px
- Inputs: 6px

### Shadows
- Cards: 0 4px 12px rgba(0,0,0,0.3)
- Dropdowns: 0 8px 24px rgba(0,0,0,0.4)

---

## Interactive States

### Buttons
```
Default:  bg-#21262D  text-#E6EDF3
Hover:    bg-#30363D  text-#FFFFFF
Active:   bg-#58A6FF  text-#FFFFFF
Disabled: bg-#161B22  text-#484F58
```

### Sidebar Icons
```
Default:  #8B949E
Hover:    #E6EDF3
Active:   #58A6FF + left border
```

---

## Responsive Breakpoints

| Breakpoint | Layout                          |
|------------|----------------------------------|
| < 768px    | Sidebar hidden, bottom nav      |
| 768-1024   | Collapsed sidebar, stacked panels |
| > 1024     | Full layout as designed         |

---

## Animations

1. **Page transitions**: 200ms ease-out slide
2. **3D model load**: Fade in with 300ms
3. **Metrics update**: Number counter animation
4. **Attention maps**: Pulsing opacity overlay
