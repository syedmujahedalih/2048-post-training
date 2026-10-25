import { Audio } from "@remotion/media";
import { staticFile, useVideoConfig } from "remotion";
import { BasicCaptions } from "../BasicCaptions";

export const RelativeNarration = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Audio
        name="Reward becomes relative advantage narration"
        src={staticFile("narration/Relative.mp3")}
        premountFor={fps}
      />
      <BasicCaptions
        width={1696}
        style={{ position: "absolute", bottom: 55, left: 112 }}
        captions={[
          {
            text: "Use a small numerical example: four rewards are zero, one, two, and three.",
            startMs: 450,
            endMs: 5922,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Their mean is one point five. Subtract that mean from each reward.",
            startMs: 6082,
            endMs: 9738,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Attempts below the mean get negative advantages. Attempts above it get positive advantages.",
            startMs: 9898,
            endMs: 15282,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "A common GRPO version also divides by the group's standard deviation.",
            startMs: 15442,
            endMs: 19711,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "The group provides the baseline, replacing a separately learned value critic.",
            startMs: 19871,
            endMs: 24144,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Exact normalization depends on the implementation and configuration.",
            startMs: 24304,
            endMs: 28341,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
        ]}
      />
    </>
  );
};
