import { Interactive, interpolate, useCurrentFrame } from "remotion";
import { GuardrailsNarration } from "../narration/GuardrailsNarration";
import {
  Card,
  Note,
  SceneFrame,
  Tag,
  sceneSchema,
  type SceneProps,
} from "../Visuals";

const GuardrailsInner = ({
  title = "Keep an update from overreacting",
  style,
}: SceneProps) => {
  const frame = useCurrentFrame();
  return (
    <SceneFrame number="08 / 11" title={title} style={style}>
      <div style={{ position: "absolute", left: 112, top: 280, width: 940 }}>
        <Tag>Clipped objective / positive A / ε = 0.2</Tag>
        <svg
          width="940"
          height="290"
          viewBox="0 0 940 330"
          style={{ overflow: "visible" }}
        >
          <line
            x1="75"
            y1="275"
            x2="875"
            y2="275"
            stroke="#9BA4A8"
            strokeWidth="3"
          />
          <line
            x1="75"
            y1="275"
            x2="75"
            y2="30"
            stroke="#9BA4A8"
            strokeWidth="3"
          />
          <rect
            x="475"
            y="30"
            width="200"
            height="245"
            fill="#2855E8"
            opacity="0.07"
          />
          <path
            d="M75 275 L675 75 L875 75"
            stroke="#2855E8"
            strokeWidth="8"
            fill="none"
          />
          <path
            d="M675 75 L875 8"
            stroke="#C0C5CD"
            strokeWidth="5"
            strokeDasharray="10 9"
            fill="none"
          />
          <text x="440" y="320" fontSize="30" fill="#687582">
            1.0
          </text>
          <text x="650" y="320" fontSize="30" fill="#2855E8">
            1.2
          </text>
          <text x="840" y="320" fontSize="30" fill="#687582">
            1.4
          </text>
        </svg>
        <div style={{ fontSize: 33, color: "#687582", textAlign: "center" }}>
          Probability ratio: current / rollout policy
        </div>
      </div>
      <Card
        style={{
          position: "absolute",
          left: 1170,
          top: 280,
          width: 638,
          height: 365,
        }}
      >
        <Tag>Optional KL penalty</Tag>
        <div style={{ fontSize: 51, fontWeight: 600, marginTop: 29 }}>
          Stay near a<br />
          reference policy
        </div>
        <div style={{ fontSize: 32, color: "#687582", marginTop: 25 }}>
          Reference anchor and
          <br />
          rollout policy have different jobs.
        </div>
      </Card>
      <Interactive.Div
        name="Clipping limitation"
        style={{
          position: "absolute",
          left: 120,
          top: 690,
          fontSize: 36,
          color: "#C05E32",
          opacity: interpolate(frame, [325, 355], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          }),
        }}
      >
        The objective flattens. The ratio can still go past the threshold.
      </Interactive.Div>
      <Note>
        Schematic: positive-advantage contribution to the surrogate objective;
        not a measured training plot.
      </Note>
      <GuardrailsNarration />
    </SceneFrame>
  );
};
export const GuardrailsScene = Interactive.withSchema({
  Component: GuardrailsInner,
  componentName: "<GuardrailsScene>",
  schema: sceneSchema,
  wrapInSequence: true,
});
