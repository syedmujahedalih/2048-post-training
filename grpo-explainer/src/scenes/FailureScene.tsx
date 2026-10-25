import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { FailureNarration } from "../narration/FailureNarration";
import {
  Card,
  Note,
  SceneFrame,
  Tag,
  sceneSchema,
  type SceneProps,
} from "../Visuals";

const FailureInner = ({
  title = "Three things to watch",
  style,
}: SceneProps) => {
  const frame = useCurrentFrame();
  return (
    <SceneFrame number="09 / 11" title={title} style={style}>
      <Card
        style={{
          position: "absolute",
          left: 112,
          top: 295,
          width: 540,
          height: 385,
        }}
      >
        <Tag>01 / No variation</Tag>
        <div style={{ fontSize: 63, fontWeight: 600, marginTop: 36 }}>
          1, 1, 1, 1
        </div>
        <div style={{ fontSize: 43, lineHeight: 1.25, marginTop: 28 }}>
          Equal rewards
          <br />→ zero relative signal
        </div>
      </Card>
      <Interactive.Div
        name="Randomness risk"
        style={{
          position: "absolute",
          left: 690,
          top: 295,
          width: 540,
          height: 385,
          opacity: interpolate(frame, [160, 190], [0.15, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <Card style={{ height: 385 }}>
          <Tag>02 / Chance</Tag>
          <div style={{ fontSize: 63, fontWeight: 600, marginTop: 36 }}>
            Lucky tiles
          </div>
          <div style={{ fontSize: 43, lineHeight: 1.25, marginTop: 28 }}>
            A good outcome can
            <br />
            come from a bad plan.
          </div>
        </Card>
      </Interactive.Div>
      <Interactive.Div
        name="Reward proxy risk"
        style={{
          position: "absolute",
          left: 1268,
          top: 295,
          width: 540,
          height: 385,
          opacity: interpolate(frame, [310, 340], [0.15, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <Card style={{ height: 385 }}>
          <Tag>03 / Proxy</Tag>
          <div style={{ fontSize: 63, fontWeight: 600, marginTop: 36 }}>
            Reward ↑
          </div>
          <div style={{ fontSize: 43, lineHeight: 1.25, marginTop: 28 }}>
            Full-game quality
            <br />
            might still fall.
          </div>
        </Card>
      </Interactive.Div>
      <Note>
        Track reward variation and behavior. Evaluate complete games on separate
        seeds.
      </Note>
      <FailureNarration />
    </SceneFrame>
  );
};
export const FailureScene = Interactive.withSchema({
  Component: FailureInner,
  componentName: "<FailureScene>",
  schema: sceneSchema,
  wrapInSequence: true,
});
