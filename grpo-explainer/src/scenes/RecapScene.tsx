import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { RecapNarration } from "../narration/RecapNarration";
import { SceneFrame, sceneSchema, type SceneProps } from "../Visuals";

const RecapInner = ({
  title = "Three ideas to remember",
  style,
}: SceneProps) => {
  const frame = useCurrentFrame();
  return (
    <SceneFrame number="11 / 11" title={title} style={style}>
      <Interactive.Div
        name="Online takeaway"
        style={{
          position: "absolute",
          left: 112,
          top: 285,
          fontSize: 50,
          opacity: interpolate(frame, [50, 80], [0.2, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: 495,
            fontWeight: 700,
            color: "#2855E8",
          }}
        >
          Online
        </span>
        Fresh gameplay from the current policy
      </Interactive.Div>
      <Interactive.Div
        name="Group relative takeaway"
        style={{
          position: "absolute",
          left: 112,
          top: 420,
          fontSize: 50,
          opacity: interpolate(frame, [180, 210], [0.2, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: 495,
            fontWeight: 700,
            color: "#2855E8",
          }}
        >
          Group relative
        </span>
        Compare attempts from one starting board
      </Interactive.Div>
      <Interactive.Div
        name="Optimization takeaway"
        style={{
          position: "absolute",
          left: 112,
          top: 555,
          fontSize: 50,
          opacity: interpolate(frame, [310, 340], [0.2, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: 495,
            fontWeight: 700,
            color: "#2855E8",
          }}
        >
          Policy optimization
        </span>
        Advantages → optimizer → changed weights
      </Interactive.Div>
      <Interactive.Div
        name="Replay prompt"
        style={{
          position: "absolute",
          left: 112,
          top: 720,
          fontSize: 34,
          color: "#687582",
          opacity: interpolate(frame, [670, 700], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Replay any chapter from the Chapters folder in the video preview.
      </Interactive.Div>
      <RecapNarration />
    </SceneFrame>
  );
};
export const RecapScene = Interactive.withSchema({
  Component: RecapInner,
  componentName: "<RecapScene>",
  schema: sceneSchema,
  wrapInSequence: true,
});
