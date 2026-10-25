import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { PiecesNarration } from "../narration/PiecesNarration";
import {
  Arrow,
  Board,
  Card,
  Note,
  SceneFrame,
  Tag,
  sceneSchema,
  type SceneProps,
} from "../Visuals";
import data from "../rollouts.json";

const PiecesInner = ({
  title = "Board. Policy. Action. Reward.",
  style,
}: SceneProps) => {
  const frame = useCurrentFrame();
  return (
    <SceneFrame number="02 / 11" title={title} style={style}>
      <div style={{ position: "absolute", top: 280, left: 112 }}>
        <Tag>State s</Tag>
        <div style={{ marginTop: 22 }}>
          <Board cells={data.start} size={360} />
        </div>
      </div>
      <div style={{ position: "absolute", top: 462, left: 510 }}>
        <Arrow />
      </div>
      <Card
        style={{
          position: "absolute",
          top: 350,
          left: 630,
          width: 490,
          height: 300,
        }}
      >
        <Tag>Policy πθ</Tag>
        <div style={{ fontSize: 52, fontWeight: 600, marginTop: 20 }}>
          Choose a move
        </div>
        <div
          style={{
            fontFamily: "Menlo, monospace",
            fontSize: 37,
            color: "#2855E8",
            marginTop: 30,
          }}
        >
          move("LEFT")
        </div>
      </Card>
      <div style={{ position: "absolute", top: 462, left: 1162 }}>
        <Arrow />
      </div>
      <Interactive.Div
        name="Next state"
        style={{
          position: "absolute",
          top: 280,
          left: 1335,
          opacity: interpolate(frame, [290, 325], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        <Tag>Next state s′</Tag>
        <div style={{ marginTop: 22 }}>
          <Board cells={data.left.board} size={360} />
        </div>
        <div style={{ fontSize: 36, color: "#12876E", marginTop: 12 }}>
          Score gained: +{data.left.gain}
        </div>
      </Interactive.Div>
      <Note>
        A rollout is a sequence of state → action → feedback interactions.
      </Note>
      <PiecesNarration />
    </SceneFrame>
  );
};
export const PiecesScene = Interactive.withSchema({
  Component: PiecesInner,
  componentName: "<PiecesScene>",
  schema: sceneSchema,
  wrapInSequence: true,
});
