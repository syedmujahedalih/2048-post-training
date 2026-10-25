import { Series, useVideoConfig } from "remotion";
import { IntroScene } from "./scenes/IntroScene";
import { PiecesScene } from "./scenes/PiecesScene";
import { OnlineScene } from "./scenes/OnlineScene";
import { GroupScene } from "./scenes/GroupScene";
import { RewardsScene } from "./scenes/RewardsScene";
import { RelativeScene } from "./scenes/RelativeScene";
import { UpdateScene } from "./scenes/UpdateScene";
import { GuardrailsScene } from "./scenes/GuardrailsScene";
import { FailureScene } from "./scenes/FailureScene";
import { ProjectScene } from "./scenes/ProjectScene";
import { RecapScene } from "./scenes/RecapScene";

export const Explainer = () => {
  const { fps } = useVideoConfig();
  return (
    <Series>
      <Series.Sequence
        name="How playing changes a model"
        durationInFrames={721}
        premountFor={fps}
      >
        <IntroScene />
      </Series.Sequence>
      <Series.Sequence
        name="Board, policy, action, reward"
        durationInFrames={707}
        premountFor={fps}
      >
        <PiecesScene />
      </Series.Sequence>
      <Series.Sequence
        name="What makes the learning online?"
        durationInFrames={701}
        premountFor={fps}
      >
        <OnlineScene />
      </Series.Sequence>
      <Series.Sequence
        name="Start together. Explore differently."
        durationInFrames={739}
        premountFor={fps}
      >
        <GroupScene />
      </Series.Sequence>
      <Series.Sequence
        name="Choose what earns reward"
        durationInFrames={760}
        premountFor={fps}
      >
        <RewardsScene />
      </Series.Sequence>
      <Series.Sequence
        name="Reward becomes relative advantage"
        durationInFrames={878}
        premountFor={fps}
      >
        <RelativeScene />
      </Series.Sequence>
      <Series.Sequence
        name="Advantage steers the policy"
        durationInFrames={824}
        premountFor={fps}
      >
        <UpdateScene />
      </Series.Sequence>
      <Series.Sequence
        name="Keep an update from overreacting"
        durationInFrames={810}
        premountFor={fps}
      >
        <GuardrailsScene />
      </Series.Sequence>
      <Series.Sequence
        name="Three things to watch"
        durationInFrames={780}
        premountFor={fps}
      >
        <FailureScene />
      </Series.Sequence>
      <Series.Sequence
        name="Our first TRL experiment"
        durationInFrames={961}
        premountFor={fps}
      >
        <ProjectScene />
      </Series.Sequence>
      <Series.Sequence
        name="Three ideas to remember"
        durationInFrames={876}
        premountFor={fps}
      >
        <RecapScene />
      </Series.Sequence>
    </Series>
  );
};
