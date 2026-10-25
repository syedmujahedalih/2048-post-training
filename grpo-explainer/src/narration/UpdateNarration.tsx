import { Audio } from "@remotion/media";
import { staticFile, useVideoConfig } from "remotion";
import { BasicCaptions } from "../BasicCaptions";

export const UpdateNarration = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Audio
        name="Advantage steers the policy narration"
        src={staticFile("narration/Update.mp3")}
        premountFor={fps}
      />
      <BasicCaptions
        width={1696}
        style={{ position: "absolute", bottom: 55, left: 112 }}
        captions={[
          {
            text: "Now the optimizer uses those advantages to change the model's probabilities.",
            startMs: 450,
            endMs: 4834,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Positive advantages encourage the sampled action tokens. Negative advantages discourage them.",
            startMs: 4994,
            endMs: 10508,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "With an episode level reward, many decisions share the same signal.",
            startMs: 10668,
            endMs: 14606,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "That does not prove every move in a successful rollout was good.",
            startMs: 14766,
            endMs: 17946,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "This is the credit assignment challenge: connecting a later outcome to earlier choices.",
            startMs: 18106,
            endMs: 22899,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "In our project, the LoRA adapters are the trainable weights.",
            startMs: 23059,
            endMs: 26527,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
        ]}
      />
    </>
  );
};
