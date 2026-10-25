import { Composition, Folder } from "remotion";
import { Explainer } from "./Main";
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

export const RemotionRoot = () => (
  <>
    <Composition
      id="GRPO-2048"
      component={Explainer}
      width={1920}
      height={1080}
      fps={30}
      durationInFrames={8757}
    />
    <Folder name="Chapters">
      <Composition
        id="Intro"
        component={IntroScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={721}
      />
      <Composition
        id="Pieces"
        component={PiecesScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={707}
      />
      <Composition
        id="Online"
        component={OnlineScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={701}
      />
      <Composition
        id="Group"
        component={GroupScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={739}
      />
      <Composition
        id="Rewards"
        component={RewardsScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={760}
      />
      <Composition
        id="Relative"
        component={RelativeScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={878}
      />
      <Composition
        id="Update"
        component={UpdateScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={824}
      />
      <Composition
        id="Guardrails"
        component={GuardrailsScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={810}
      />
      <Composition
        id="Failure"
        component={FailureScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={780}
      />
      <Composition
        id="Project"
        component={ProjectScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={961}
      />
      <Composition
        id="Recap"
        component={RecapScene}
        width={1920}
        height={1080}
        fps={30}
        durationInFrames={876}
      />
    </Folder>
  </>
);
