import { Audio } from "@remotion/media";
import { staticFile, useVideoConfig } from "remotion";
import { BasicCaptions } from "../BasicCaptions";

export const PiecesNarration = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Audio
        name="Board, policy, action, reward narration"
        src={staticFile("narration/Pieces.mp3")}
        premountFor={fps}
      />
      <BasicCaptions
        width={1696}
        style={{ position: "absolute", bottom: 55, left: 112 }}
        captions={[
          {
            text: "The board is the state. Our model reads a text version of that board.",
            startMs: 450,
            endMs: 4096,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Its policy describes how likely it is to choose each possible move.",
            startMs: 4256,
            endMs: 7922,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "In our language model, those choices are produced through the tokens of a move tool call.",
            startMs: 8082,
            endMs: 12641,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "The simulator applies the move, merges tiles, adds a random tile, and returns a new board.",
            startMs: 12801,
            endMs: 18595,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "A sequence of these interactions is called a rollout, or trajectory.",
            startMs: 18755,
            endMs: 22630,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
        ]}
      />
    </>
  );
};
