import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { RelativeNarration } from "../narration/RelativeNarration";
import {
  Note,
  SceneFrame,
  Tag,
  sceneSchema,
  type SceneProps,
} from "../Visuals";

const RelativeInner = ({
  title = "Reward → relative advantage",
  style,
}: SceneProps) => {
  const frame = useCurrentFrame();
  return (
    <SceneFrame number="06 / 11" title={title} style={style}>
      <div style={{ position: "absolute", left: 112, top: 270 }}>
        <Tag>Toy rewards / mean = 1.5</Tag>
      </div>
      <div
        style={{
          position: "absolute",
          left: 112,
          top: 335,
          width: 1696,
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 25,
        }}
      >
        {[
          { r: 0, a: "−1.34", centered: "−1.5", positive: false },
          { r: 1, a: "−0.45", centered: "−0.5", positive: false },
          { r: 2, a: "+0.45", centered: "+0.5", positive: true },
          { r: 3, a: "+1.34", centered: "+1.5", positive: true },
        ].map(({ r, a, centered, positive }, i) => (
          <div
            key={i}
            style={{
              backgroundColor: "#FFFFFF",
              borderRadius: 22,
              padding: "25px 30px",
              height: 244,
              boxSizing: "border-box",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: 72, fontWeight: 600 }}>{r}</div>
            <div style={{ fontSize: 27, color: "#687582" }}>reward</div>
            <div
              style={{
                opacity: interpolate(frame, [185, 210], [0, 1], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                }),
                color: positive ? "#12876E" : "#C05E32",
                fontSize: 57,
                fontWeight: 600,
                marginTop: 13,
              }}
            >
              {frame < 475 ? centered : a}
            </div>
          </div>
        ))}
      </div>
      <Interactive.Div
        name="Normalized advantage"
        style={{
          position: "absolute",
          left: 120,
          top: 626,
          fontSize: 54,
          fontWeight: 600,
          opacity: interpolate(frame, [465, 490], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Aᵢ = (rᵢ − group mean) / (group std + ε)
      </Interactive.Div>
      <Note>
        Illustration uses population std ≈ 1.118. Exact normalization depends on
        TRL configuration.
      </Note>
      <RelativeNarration />
    </SceneFrame>
  );
};
export const RelativeScene = Interactive.withSchema({
  Component: RelativeInner,
  componentName: "<RelativeScene>",
  schema: sceneSchema,
  wrapInSequence: true,
});
