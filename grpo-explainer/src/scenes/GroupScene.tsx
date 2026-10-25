import { Interactive, useCurrentFrame } from "remotion";
import { GroupNarration } from "../narration/GroupNarration";
import {
  Board,
  Note,
  SceneFrame,
  Tag,
  sceneSchema,
  type SceneProps,
} from "../Visuals";
import data from "../rollouts.json";

const GroupInner = ({
  title = "Start together. Explore differently.",
  style,
}: SceneProps) => {
  const frame = useCurrentFrame();
  const step = Math.min(16, Math.max(0, Math.floor((frame - 150) / 25)));
  return (
    <SceneFrame number="04 / 11" title={title} style={style}>
      <Interactive.Div
        name="Group definition"
        style={{
          position: "absolute",
          left: 112,
          top: 244,
          fontSize: 39,
          color: "#2855E8",
        }}
      >
        Group Relative Policy Optimization · four attempts from one starting
        board
      </Interactive.Div>
      <div
        style={{
          position: "absolute",
          left: 112,
          top: 326,
          display: "flex",
          gap: 158,
        }}
      >
        {data.rollouts.map((rollout, index) => {
          const current = Math.min(step, rollout.length - 1);
          return (
            <div key={index}>
              <Tag>Rollout {index + 1}</Tag>
              <div style={{ marginTop: 20 }}>
                <Board cells={rollout[current].board} size={300} />
              </div>
              <div
                style={{
                  fontSize: 33,
                  marginTop: 14,
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {current < step ? "Done at" : "Step"} {current}/16 · +
                {rollout[current].score}
              </div>
            </div>
          );
        })}
      </div>
      <Note>
        Illustrative simulator rollouts using uniform legal moves; these are not
        trained model results.
      </Note>
      <GroupNarration />
    </SceneFrame>
  );
};
export const GroupScene = Interactive.withSchema({
  Component: GroupInner,
  componentName: "<GroupScene>",
  schema: sceneSchema,
  wrapInSequence: true,
});
