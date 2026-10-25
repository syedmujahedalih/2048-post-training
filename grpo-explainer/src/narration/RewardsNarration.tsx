import { Audio } from "@remotion/media";
import { staticFile, useVideoConfig } from "remotion";
import { BasicCaptions } from "../BasicCaptions";

export const RewardsNarration = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Audio
        name="Choose what earns reward narration"
        src={staticFile("narration/Rewards.mp3")}
        premountFor={fps}
      />
      <BasicCaptions
        width={1696}
        style={{ position: "absolute", bottom: 55, left: 112 }}
        captions={[
          {
            text: "A reward function converts each outcome into a number.",
            startMs: 450,
            endMs: 3268,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "For 2048, we can compute feedback directly from the simulator.",
            startMs: 3428,
            endMs: 7748,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Our initial proposal uses the logarithm of one plus score gained, minus a penalty for invalid attempts.",
            startMs: 7908,
            endMs: 13977,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "That rewards progress during a short episode, but it is only a proxy for strong full game play.",
            startMs: 14137,
            endMs: 19626,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "We still need held out games to check whether optimizing that proxy actually helps.",
            startMs: 19786,
            endMs: 24421,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
        ]}
      />
    </>
  );
};
