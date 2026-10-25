import { Audio } from "@remotion/media";
import { staticFile, useVideoConfig } from "remotion";
import { BasicCaptions } from "../BasicCaptions";

export const GuardrailsNarration = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Audio
        name="Keep an update from overreacting narration"
        src={staticFile("narration/Guardrails.mp3")}
        premountFor={fps}
      />
      <BasicCaptions
        width={1696}
        style={{ position: "absolute", bottom: 55, left: 112 }}
        captions={[
          {
            text: "GRPO can use a clipped objective to limit the incentive for overly large probability changes.",
            startMs: 450,
            endMs: 5959,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "For a positive advantage, increasing the probability ratio beyond the clipping threshold stops improving that part of the objective.",
            startMs: 6119,
            endMs: 13399,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Clipping is not a hard guarantee that the probability ratio stays inside the band.",
            startMs: 13559,
            endMs: 17853,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "An optional KL penalty can also anchor the policy to a reference model.",
            startMs: 18013,
            endMs: 22224,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "The old rollout policy and the reference policy have different jobs.",
            startMs: 22384,
            endMs: 26086,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
        ]}
      />
    </>
  );
};
