import { Audio } from "@remotion/media";
import { staticFile, useVideoConfig } from "remotion";
import { BasicCaptions } from "../BasicCaptions";

export const IntroNarration = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Audio
        name="How playing changes a model narration"
        src={staticFile("narration/Intro.mp3")}
        premountFor={fps}
      />
      <BasicCaptions
        width={1696}
        style={{ position: "absolute", bottom: 55, left: 112 }}
        captions={[
          {
            text: "You already know how to play 2048. Now imagine teaching a language model to play it.",
            startMs: 450,
            endMs: 5922,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Supervised fine tuning gives it examples of a solver's moves.",
            startMs: 6082,
            endMs: 9661,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Online reinforcement learning lets it play, receive feedback, and change its future behavior.",
            startMs: 9821,
            endMs: 15563,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "GRPO is one way to turn that feedback into a training signal.",
            startMs: 15723,
            endMs: 19374,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Let's follow one training iteration, from a board to an update.",
            startMs: 19534,
            endMs: 23090,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
        ]}
      />
    </>
  );
};
