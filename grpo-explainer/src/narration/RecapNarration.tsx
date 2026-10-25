import { Audio } from "@remotion/media";
import { staticFile, useVideoConfig } from "remotion";
import { BasicCaptions } from "../BasicCaptions";

export const RecapNarration = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Audio
        name="Three ideas to remember narration"
        src={staticFile("narration/Recap.mp3")}
        premountFor={fps}
      />
      <BasicCaptions
        width={1696}
        style={{ position: "absolute", bottom: 55, left: 112 }}
        captions={[
          {
            text: "Before starting the GPU run, check these three ideas.",
            startMs: 450,
            endMs: 3752,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Online: fresh gameplay comes from the policy being trained.",
            startMs: 3912,
            endMs: 7505,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Group relative: compare attempts that share a starting board.",
            startMs: 7665,
            endMs: 10977,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Policy optimization: rewards become advantages, and optimizer steps change the trainable weights.",
            startMs: 11137,
            endMs: 17499,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "The experiment succeeds as a learning exercise when we can explain that whole loop and measure its results honestly.",
            startMs: 17659,
            endMs: 23544,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Pause, replay a chapter, and then we are ready for the first small training check.",
            startMs: 23704,
            endMs: 28288,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
        ]}
      />
    </>
  );
};
