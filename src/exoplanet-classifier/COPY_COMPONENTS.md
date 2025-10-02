# Component Copy Instructions

## 🔄 ShadCN UI Components Required

You need to copy the following UI components from the original `/components/ui/` directory to `/exoplanet-classifier/src/components/ui/`:

### Essential Components (Copy these files):

```
accordion.tsx          - Collapsible content sections
alert-dialog.tsx       - Modal dialog alerts  
alert.tsx             - Notification alerts
aspect-ratio.tsx      - Responsive aspect ratios
avatar.tsx            - User profile images
badge.tsx             - Status indicators
breadcrumb.tsx        - Navigation breadcrumbs
button.tsx            - Interactive buttons
calendar.tsx          - Date picker calendar
card.tsx              - Content containers
carousel.tsx          - Image/content carousels
chart.tsx             - Chart components wrapper
checkbox.tsx          - Checkbox inputs
collapsible.tsx       - Expandable sections
command.tsx           - Command palette
context-menu.tsx      - Right-click menus
dialog.tsx            - Modal dialogs
drawer.tsx            - Slide-out panels
dropdown-menu.tsx     - Dropdown menus
form.tsx              - Form components
hover-card.tsx        - Hover popover cards
input-otp.tsx         - OTP input fields
input.tsx             - Text input fields
label.tsx             - Form labels
menubar.tsx           - Menu bar navigation
navigation-menu.tsx   - Navigation menus
pagination.tsx        - Page navigation
popover.tsx           - Floating popover content
progress.tsx          - Progress bars
radio-group.tsx       - Radio button groups
resizable.tsx         - Resizable panels
scroll-area.tsx       - Custom scrollable areas
select.tsx            - Dropdown selectors
separator.tsx         - Visual separators
sheet.tsx             - Side panels
sidebar.tsx           - Sidebar navigation
skeleton.tsx          - Loading skeletons
slider.tsx            - Range sliders
sonner.tsx            - Toast notifications
switch.tsx            - Toggle switches
table.tsx             - Data tables
tabs.tsx              - Tabbed interfaces
textarea.tsx          - Multi-line text inputs
toggle-group.tsx      - Toggle button groups
toggle.tsx            - Toggle buttons
tooltip.tsx           - Hover tooltips
use-mobile.ts         - Mobile detection hook
utils.ts              - Utility functions
```

### Components Currently Used in App:

The application specifically uses these components:
- `button.tsx` - For all interactive buttons
- `card.tsx` - For content containers
- `input.tsx` - For feature input fields
- `label.tsx` - For form labels
- `select.tsx` - For model selection
- `tabs.tsx` - For main navigation
- `badge.tsx` - For status indicators
- `progress.tsx` - For metric displays
- `tooltip.tsx` - For feature descriptions
- `sonner.tsx` - For toast notifications

## 📋 Quick Copy Process

### Option 1: Manual Copy
1. Navigate to your original `/components/ui/` directory
2. Copy each `.tsx` and `.ts` file listed above
3. Paste into `/exoplanet-classifier/src/components/ui/`

### Option 2: Command Line (Linux/Mac)
```bash
# From the original project root
cp -r components/ui/* exoplanet-classifier/src/components/ui/
```

### Option 3: Command Line (Windows)
```cmd
# From the original project root  
xcopy components\ui\*.* exoplanet-classifier\src\components\ui\ /s
```

## ⚠️ Important Notes

1. **Utils File**: The `utils.ts` file contains the `cn` utility function used throughout
2. **Dependencies**: All components depend on Radix UI primitives (already in package.json)
3. **Tailwind Classes**: Components use Tailwind utility classes
4. **TypeScript**: All components are written in TypeScript

## 🔍 Verification

After copying, ensure these files exist:
- `/exoplanet-classifier/src/components/ui/utils.ts`
- `/exoplanet-classifier/src/components/ui/button.tsx`
- `/exoplanet-classifier/src/components/ui/card.tsx`
- `/exoplanet-classifier/src/components/ui/input.tsx`
- `/exoplanet-classifier/src/components/ui/select.tsx`
- `/exoplanet-classifier/src/components/ui/tabs.tsx`

## 🚀 Next Steps

After copying the UI components:

1. **Copy Main Components** (see COPY_MAIN_COMPONENTS.md)
2. **Install Dependencies**: `npm install`
3. **Start Development**: `npm run dev`
4. **Verify App Loads**: Check http://localhost:5173

---

This ensures all UI components are available for the exoplanet classifier application.