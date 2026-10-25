import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { ProjectNarration } from "../narration/ProjectNarration";
import {
  Arrow,
  Card,
  Note,
  SceneFrame,
  Tag,
  sceneSchema,
  type SceneProps,
} from "../Visuals";

const ProjectInner = ({
  title = "Our first TRL experiment",
  style,
}: SceneProps) => {
  const frame = useCurrentFrame();
  return (
    <SceneFrame number="10 / 11" title={title} style={style}>
      <div
        style={{
          position: "absolute",
          left: 112,
          top: 300,
          display: "flex",
          gap: 20,
          alignItems: "center",
        }}
      >
        <Card style={{ width: 320, height: 350 }}>
          <Tag>Start</Tag>
          <div
            style={{
              fontSize: 46,
              lineHeight: 1.12,
              fontWeight: 600,
              marginTop: 25,
            }}
          >
            SFT
            <br />
            checkpoint
          </div>
          <div
            style={{
              fontSize: 28,
              lineHeight: 1.2,
              color: "#687582",
              marginTop: 24,
            }}
          >
            Solver demonstrations
          </div>
        </Card>
        <Arrow />
        <Card style={{ width: 320, height: 350 }}>
          <Tag>Collect</Tag>
          <div
            style={{
              fontSize: 46,
              lineHeight: 1.12,
              fontWeight: 600,
              marginTop: 25,
            }}
          >
            4 rollouts ×<br />
            16 attempts
          </div>
          <div
            style={{
              fontSize: 28,
              lineHeight: 1.2,
              color: "#687582",
              marginTop: 24,
            }}
          >
            Student picks actions
          </div>
        </Card>
        <Arrow />
        <Card style={{ width: 350, height: 350, borderColor: "#2855E8" }}>
          <Tag>Inspect</Tag>
          <div
            style={{
              fontSize: 46,
              lineHeight: 1.12,
              fontWeight: 600,
              marginTop: 25,
            }}
          >
            10 updates
          </div>
          <div
            style={{
              fontSize: 28,
              lineHeight: 1.2,
              color: "#687582",
              marginTop: 24,
            }}
          >
            Masks · advantages
            <br />
            gradients · weights
          </div>
        </Card>
        <Arrow />
        <Card style={{ width: 320, height: 350 }}>
          <Tag>Evaluate</Tag>
          <div
            style={{
              fontSize: 46,
              lineHeight: 1.12,
              fontWeight: 600,
              marginTop: 25,
            }}
          >
            Full games
          </div>
          <div
            style={{
              fontSize: 28,
              lineHeight: 1.2,
              color: "#687582",
              marginTop: 24,
            }}
          >
            Compare with SFT alone
          </div>
        </Card>
      </div>
      <Interactive.Div
        name="Training status"
        style={{
          position: "absolute",
          left: 120,
          top: 678,
          fontSize: 40,
          color: "#C05E32",
          opacity: interpolate(frame, [750, 780], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        This is the proposed experiment. RL training has not run yet.
      </Interactive.Div>
      <Note>
        The solver teaches during SFT; RL actions must come from the student
        policy.
      </Note>
      <ProjectNarration />
    </SceneFrame>
  );
};
export const ProjectScene = Interactive.withSchema({
  Component: ProjectInner,
  componentName: "<ProjectScene>",
  schema: sceneSchema,
  wrapInSequence: true,
});
