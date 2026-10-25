import type { CSSProperties, ReactNode } from "react";
import {
  AbsoluteFill,
  Interactive,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  type InteractivitySchema,
} from "remotion";

export type SceneProps = { title?: string; style?: CSSProperties };
export const sceneSchema = {
  title: { type: "text-content", default: "", description: "Chapter heading" },
} as const satisfies InteractivitySchema;

export const SceneFrame = ({
  number,
  title,
  children,
  style,
}: {
  number: string;
  title: string;
  children: ReactNode;
  style?: CSSProperties;
}) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#F5F2EB",
        color: "#152537",
        fontFamily: "Avenir Next, Arial, sans-serif",
        ...style,
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          height: 8,
          backgroundColor: "#2855E8",
          width: interpolate(frame, [0, durationInFrames], [0, 1920], {
            extrapolateRight: "clamp",
          }),
        }}
      />
      <Interactive.Div
        name="Chapter label"
        style={{
          position: "absolute",
          left: 112,
          top: 48,
          fontSize: 29,
          fontWeight: 600,
          letterSpacing: 3,
          color: "#687582",
        }}
      >
        2048 LAB / {number}
      </Interactive.Div>
      <Interactive.Div
        name="Chapter heading"
        style={{
          position: "absolute",
          left: 112,
          top: 105,
          width: 1696,
          fontSize: 86,
          fontWeight: 700,
          letterSpacing: -3,
          lineHeight: 1.1,
          opacity: interpolate(frame, [0, 14], [0, 1], {
            extrapolateRight: "clamp",
          }),
        }}
      >
        {title}
      </Interactive.Div>
      {children}
    </AbsoluteFill>
  );
};

const tileColors: Record<number, string> = {
  0: "#DCD8CD",
  2: "#EEE8DA",
  4: "#E7DBC0",
  8: "#E9AC6A",
  16: "#E58D4F",
  32: "#DD704B",
  64: "#CA5140",
  128: "#DDBE50",
  256: "#D4B03D",
  512: "#CBA12C",
  1024: "#BE9320",
  2048: "#AE841B",
};

export const Board = ({
  cells,
  size = 420,
}: {
  cells: number[];
  size?: number;
}) => (
  <div
    style={{
      width: size,
      height: size,
      padding: size * 0.025,
      display: "grid",
      gridTemplateColumns: "repeat(4, 1fr)",
      gap: size * 0.018,
      backgroundColor: "#B7AF9E",
      borderRadius: size * 0.03,
      boxSizing: "border-box",
    }}
  >
    {cells.map((value, i) => (
      <div
        key={i}
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          backgroundColor: tileColors[value] ?? "#152537",
          borderRadius: size * 0.012,
          color: value >= 8 ? "#FFFFFF" : "#594E3D",
          fontSize: size * (value >= 1000 ? 0.085 : 0.105),
          fontWeight: 700,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value || ""}
      </div>
    ))}
  </div>
);

export const Tag = ({
  children,
  color = "#2855E8",
}: {
  children: ReactNode;
  color?: string;
}) => (
  <div
    style={{
      fontSize: 30,
      fontWeight: 600,
      color,
      letterSpacing: 1.5,
      textTransform: "uppercase",
    }}
  >
    {children}
  </div>
);

export const Note = ({ children }: { children: ReactNode }) => (
  <div
    style={{
      position: "absolute",
      left: 112,
      top: 747,
      width: 1696,
      fontSize: 29,
      color: "#687582",
    }}
  >
    {children}
  </div>
);

export const Arrow = ({ down = false }: { down?: boolean }) => (
  <div
    style={{ fontSize: 66, color: "#2855E8", fontWeight: 400, lineHeight: 1 }}
  >
    {down ? "↓" : "→"}
  </div>
);

export const Card = ({
  children,
  style,
}: {
  children: ReactNode;
  style?: CSSProperties;
}) => (
  <div
    style={{
      backgroundColor: "#FFFFFF",
      border: "2px solid #DFE0DA",
      borderRadius: 24,
      padding: 30,
      boxSizing: "border-box",
      ...style,
    }}
  >
    {children}
  </div>
);
