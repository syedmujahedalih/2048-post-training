import { Audio } from "@remotion/media";
import { staticFile, useVideoConfig } from "remotion";
import { BasicCaptions } from "../BasicCaptions";

export const FailureNarration = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Audio
        name="Three things to watch narration"
        src={staticFile("narration/Failure.mp3")}
        premountFor={fps}
      />
      <BasicCaptions
        width={1696}
        style={{ position: "absolute", bottom: 55, left: 112 }}
        captions={[
          {
            text: "First, equal rewards provide no group relative preference signal.",
            startMs: 450,
            endMs: 4364,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Second, lucky tile spawns can make a poor strategy look good in a small sample.",
            startMs: 4524,
            endMs: 9206,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Third, a model can improve the short episode reward while getting worse at complete games.",
            startMs: 9366,
            endMs: 14175,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Track reward variation and policy behavior, then evaluate on separate game seeds.",
            startMs: 14335,
            endMs: 19409,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "A rising training reward is evidence to investigate, not proof that the model learned a better strategy.",
            startMs: 19569,
            endMs: 25080,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
        ]}
      />
    </>
  );
};
