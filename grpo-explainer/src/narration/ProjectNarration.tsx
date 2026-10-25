import { Audio } from "@remotion/media";
import { staticFile, useVideoConfig } from "remotion";
import { BasicCaptions } from "../BasicCaptions";

export const ProjectNarration = () => {
  const { fps } = useVideoConfig();
  return (
    <>
      <Audio
        name="Our first TRL experiment narration"
        src={staticFile("narration/Project.mp3")}
        premountFor={fps}
      />
      <BasicCaptions
        width={1696}
        style={{ position: "absolute", bottom: 55, left: 112 }}
        captions={[
          {
            text: "Here is our plan. Begin with the supervised checkpoint trained on solver demonstrations.",
            startMs: 450,
            endMs: 5450,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Let the student play short episodes. Use the simulator's outcomes to calculate rewards.",
            startMs: 5610,
            endMs: 10920,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Inspect ten updates: the sampled actions, observation masks, advantages, gradients, and changed adapter weights.",
            startMs: 11080,
            endMs: 19018,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "Then compare against supervised fine tuning alone on held out full games.",
            startMs: 19178,
            endMs: 23182,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "The solver teaches during SFT. It does not silently replace the student's sampled actions during RL.",
            startMs: 23342,
            endMs: 29240,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
          {
            text: "We have not run that training yet.",
            startMs: 29400,
            endMs: 31097,
            timestampMs: null,
            confidence: null,
            pageBreakAfter: true,
          },
        ]}
      />
    </>
  );
};
