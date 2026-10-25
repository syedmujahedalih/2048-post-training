import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { RewardsNarration } from "../narration/RewardsNarration";
import {
  Card,
  Note,
  SceneFrame,
  Tag,
  sceneSchema,
  type SceneProps,
} from "../Visuals";

const RewardsInner = ({
  title = "Choose what earns reward",
  style,
}: SceneProps) => {
  const frame = useCurrentFrame();
  return (
    <SceneFrame number="05 / 11" title={title} style={style}>
      <Card
        style={{
          position: "absolute",
          left: 112,
          top: 287,
          width: 1696,
          height: 220,
        }}
      >
        <Tag>Initial proposal / short episode</Tag>
        <Interactive.Div
          name="Proposed reward formula"
          style={{
            fontSize: 70,
            fontWeight: 600,
            marginTop: 30,
            letterSpacing: -1,
          }}
        >
          r = log(1 + Δscore) − 0.1 × invalid attempts
        </Interactive.Div>
      </Card>
      <Interactive.Div
        name="Progress signal"
        style={{
          position: "absolute",
          left: 120,
          top: 560,
          fontSize: 45,
          color: "#12876E",
          opacity: interpolate(frame, [250, 280], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        ↑ Make progress
      </Interactive.Div>
      <Interactive.Div
        name="Invalid move penalty"
        style={{
          position: "absolute",
          left: 830,
          top: 560,
          fontSize: 45,
          color: "#C05E32",
          opacity: interpolate(frame, [330, 360], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        ↓ Waste fewer attempts
      </Interactive.Div>
      <Interactive.Div
        name="Evaluation reminder"
        style={{
          position: "absolute",
          left: 120,
          top: 655,
          fontSize: 45,
          fontWeight: 600,
          opacity: interpolate(frame, [460, 490], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Check full games separately.
      </Interactive.Div>
      <Note>
        log is the natural logarithm. This reward is a proxy for the behavior we
        want.
      </Note>
      <RewardsNarration />
    </SceneFrame>
  );
};
export const RewardsScene = Interactive.withSchema({
  Component: RewardsInner,
  componentName: "<RewardsScene>",
  schema: sceneSchema,
  wrapInSequence: true,
});
