import { Audio } from "@remotion/media";
import { staticFile, useVideoConfig } from "remotion";
import { BasicCaptions } from "../BasicCaptions";

export const GroupNarration = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Audio
        name="Start together. Explore differently. narration"
        src={staticFile("narration/Group.mp3")}
        premountFor={fps}
      />
      <BasicCaptions
        width={1696}
        style={{ position: "absolute", bottom: 55, left: 112 }}
        captions={[
          {
            text: "GRPO stands for Group Relative Policy Optimization.",
            startMs: 450,
            endMs: 4066,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "For one starting board, sample several rollouts from the same policy.",
            startMs: 4226,
            endMs: 8245,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Our first experiment proposes four rollouts, each capped at sixteen decision attempts.",
            startMs: 8405,
            endMs: 13602,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "They can choose different moves, and random tile spawns can produce different outcomes.",
            startMs: 13762,
            endMs: 18509,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "This group supplies a local comparison: how did each attempt perform against its siblings?",
            startMs: 18669,
            endMs: 23706,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
        ]}
      />
    </>
  );
};
