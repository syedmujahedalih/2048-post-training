import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { OnlineNarration } from "../narration/OnlineNarration";
import {
  Arrow,
  Card,
  Note,
  SceneFrame,
  Tag,
  sceneSchema,
  type SceneProps,
} from "../Visuals";

const OnlineInner = ({
  title = "Online = fresh experience",
  style,
}: SceneProps) => {
  const frame = useCurrentFrame();
  return (
    <SceneFrame number="03 / 11" title={title} style={style}>
      <div
        style={{
          position: "absolute",
          top: 300,
          left: 112,
          display: "flex",
          gap: 35,
          alignItems: "center",
        }}
      >
        <Card style={{ width: 345, height: 350 }}>
          <Tag>1 / Sample</Tag>
          <div
            style={{
              fontSize: 51,
              lineHeight: 1.12,
              fontWeight: 600,
              marginTop: 25,
            }}
          >
            Play a batch
          </div>
          <div style={{ fontSize: 33, marginTop: 16, color: "#687582" }}>
            Policy πk is fixed
          </div>
        </Card>
        <Arrow />
        <Card style={{ width: 300, height: 350 }}>
          <Tag>2 / Score</Tag>
          <div
            style={{
              fontSize: 51,
              lineHeight: 1.12,
              fontWeight: 600,
              marginTop: 25,
            }}
          >
            Rewards
          </div>
          <div style={{ fontSize: 33, marginTop: 16, color: "#687582" }}>
            From the simulator
          </div>
        </Card>
        <Arrow />
        <Interactive.Div
          name="Optimizer changes weights"
          style={{
            width: 350,
            height: 350,
            borderRadius: 24,
            padding: 30,
            boxSizing: "border-box",
            backgroundColor: "#2855E8",
            color: "white",
            opacity: interpolate(frame, [310, 340], [0.25, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }),
          }}
        >
          <div style={{ fontSize: 30, letterSpacing: 1.5 }}>3 / OPTIMIZE</div>
          <div
            style={{
              fontSize: 51,
              lineHeight: 1.12,
              fontWeight: 600,
              marginTop: 25,
            }}
          >
            Update θ
          </div>
          <div style={{ fontSize: 33, marginTop: 16 }}>Trainable weights</div>
        </Interactive.Div>
        <Arrow />
        <Card style={{ width: 290, height: 350 }}>
          <Tag>4 / Repeat</Tag>
          <div
            style={{
              fontSize: 51,
              lineHeight: 1.12,
              fontWeight: 600,
              marginTop: 25,
            }}
          >
            Policy πk+1
          </div>
          <div style={{ fontSize: 33, marginTop: 16, color: "#687582" }}>
            New gameplay
          </div>
        </Card>
      </div>
      <Note>Playing gathers data. The optimizer step changes the model.</Note>
      <OnlineNarration />
    </SceneFrame>
  );
};
export const OnlineScene = Interactive.withSchema({
  Component: OnlineInner,
  componentName: "<OnlineScene>",
  schema: sceneSchema,
  wrapInSequence: true,
});
