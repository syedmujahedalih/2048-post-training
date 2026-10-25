import { Audio } from "@remotion/media";
import { staticFile, useVideoConfig } from "remotion";
import { BasicCaptions } from "../BasicCaptions";

export const OnlineNarration = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Audio
        name="What makes the learning online? narration"
        src={staticFile("narration/Online.mp3")}
        premountFor={fps}
      />
      <BasicCaptions
        width={1696}
        style={{ position: "absolute", bottom: 55, left: 112 }}
        captions={[
          {
            text: "Online means that training keeps collecting fresh experience from the policy being trained.",
            startMs: 450,
            endMs: 5182,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "It does not mean the model needs to browse the internet.",
            startMs: 5342,
            endMs: 8020,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "First, hold a snapshot of the policy fixed while it generates a batch of gameplay.",
            startMs: 8180,
            endMs: 13162,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Then score those rollouts and run optimizer updates on the trainable weights.",
            startMs: 13322,
            endMs: 17372,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "The updated policy generates the next batch. Playing alone does not change the weights.",
            startMs: 17532,
            endMs: 22442,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
        ]}
      />
    </>
  );
};
