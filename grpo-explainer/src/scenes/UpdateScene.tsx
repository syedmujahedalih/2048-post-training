import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { UpdateNarration } from "../narration/UpdateNarration";
import {
  Card,
  Note,
  SceneFrame,
  Tag,
  sceneSchema,
  type SceneProps,
} from "../Visuals";

const UpdateInner = ({
  title = "Advantage steers probabilities",
  style,
}: SceneProps) => {
  const frame = useCurrentFrame();
  return (
    <SceneFrame number="07 / 11" title={title} style={style}>
      <Card
        style={{
          position: "absolute",
          left: 112,
          top: 290,
          width: 805,
          height: 330,
        }}
      >
        <Tag color="#12876E">Positive advantage</Tag>
        <div
          style={{
            fontSize: 62,
            marginTop: 30,
            fontWeight: 600,
            color: "#12876E",
          }}
        >
          Encourage these
          <br />
          sampled action tokens ↑
        </div>
      </Card>
      <Card
        style={{
          position: "absolute",
          left: 955,
          top: 290,
          width: 853,
          height: 330,
        }}
      >
        <Tag color="#C05E32">Negative advantage</Tag>
        <div
          style={{
            fontSize: 62,
            marginTop: 30,
            fontWeight: 600,
            color: "#C05E32",
          }}
        >
          Discourage these
          <br />
          sampled action tokens ↓
        </div>
      </Card>
      <Interactive.Div
        name="Trainable adapters"
        style={{
          position: "absolute",
          left: 120,
          top: 659,
          fontSize: 43,
          fontWeight: 600,
          opacity: interpolate(frame, [660, 690], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        In our project: optimizer → LoRA adapter weights
      </Interactive.Div>
      <Note>
        Episode reward is a shared signal; it does not identify which individual
        moves were good.
      </Note>
      <UpdateNarration />
    </SceneFrame>
  );
};
export const UpdateScene = Interactive.withSchema({
  Component: UpdateInner,
  componentName: "<UpdateScene>",
  schema: sceneSchema,
  wrapInSequence: true,
});
