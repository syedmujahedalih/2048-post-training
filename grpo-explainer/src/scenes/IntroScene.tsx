import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { IntroNarration } from "../narration/IntroNarration";
import {
  Board,
  SceneFrame,
  Tag,
  sceneSchema,
  type SceneProps,
} from "../Visuals";
import data from "../rollouts.json";

const IntroInner = ({
  title = "How playing changes a model",
  style,
}: SceneProps) => {
  const frame = useCurrentFrame();
  const step = Math.min(16, Math.max(0, Math.floor((frame - 70) / 30)));
  return (
    <SceneFrame number="01 / 11" title={title} style={style}>
      <Interactive.Div
        name="GRPO title"
        style={{
          position: "absolute",
          left: 112,
          top: 265,
          fontSize: 170,
          letterSpacing: -7,
          fontWeight: 700,
          color: "#2855E8",
        }}
      >
        GRPO
      </Interactive.Div>
      <Interactive.Div
        name="Central question"
        style={{
          position: "absolute",
          left: 120,
          top: 464,
          width: 990,
          fontSize: 60,
          fontWeight: 500,
          lineHeight: 1.2,
        }}
      >
        Can playing change
        <br />
        the model’s weights?
      </Interactive.Div>
      <Interactive.Div
        name="Learning sequence"
        style={{
          position: "absolute",
          left: 120,
          top: 659,
          fontSize: 36,
          color: "#687582",
          opacity: interpolate(frame, [170, 200], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        Examples → Experience → Updates
      </Interactive.Div>
      <div style={{ position: "absolute", left: 1278, top: 269 }}>
        <Board cells={data.rollouts[1][step].board} size={426} />
        <div style={{ marginTop: 18 }}>
          <Tag>Our 2048 learning lab</Tag>
        </div>
      </div>
      <IntroNarration />
    </SceneFrame>
  );
};
export const IntroScene = Interactive.withSchema({
  Component: IntroInner,
  componentName: "<IntroScene>",
  schema: sceneSchema,
  wrapInSequence: true,
});
