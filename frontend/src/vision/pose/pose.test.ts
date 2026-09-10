import { describe, expect, it } from "vitest";

import type { Detection } from "../draw";
import { cropBox, personBoxes, scalePeople, toSourceKeypoints } from "./pose";
import type { Person } from "./skeleton";

const det = (
  label: string,
  score: number,
  box: [number, number, number, number],
): Detection => ({
  label,
  score,
  box: { xmin: box[0], ymin: box[1], xmax: box[2], ymax: box[3] },
});

describe("personBoxes", () => {
  it("keeps only the person class", () => {
    // Running a pose model on a car crop is wasted work and a confusing
    // overlay — and on a street scene most detections are not people.
    const out = personBoxes(
      [
        det("car", 0.99, [0, 0, 10, 10]),
        det("person", 0.8, [10, 0, 20, 20]),
        det("traffic light", 0.9, [5, 5, 6, 6]),
      ],
      { threshold: 0.4, maxPeople: 5 },
    );
    expect(out.map((d) => d.label)).toEqual(["person"]);
  });

  it("drops people below the threshold", () => {
    const out = personBoxes(
      [det("person", 0.8, [0, 0, 1, 1]), det("person", 0.2, [1, 1, 2, 2])],
      { threshold: 0.4, maxPeople: 5 },
    );
    expect(out).toHaveLength(1);
  });

  it("caps at the most confident, because each extra person is a forward pass", () => {
    const out = personBoxes(
      [
        det("person", 0.5, [0, 0, 1, 1]),
        det("person", 0.9, [1, 1, 2, 2]),
        det("person", 0.7, [2, 2, 3, 3]),
      ],
      { threshold: 0.1, maxPeople: 2 },
    );
    expect(out.map((d) => d.score)).toEqual([0.9, 0.7]);
  });

  it("returns nothing for a null result or a zero cap", () => {
    expect(personBoxes(null, { threshold: 0.4, maxPeople: 5 })).toEqual([]);
    expect(
      personBoxes([det("person", 0.9, [0, 0, 1, 1])], {
        threshold: 0.4,
        maxPeople: 0,
      }),
    ).toEqual([]);
  });
});

describe("cropBox", () => {
  it("converts a detection box to a COCO [x, y, width, height]", () => {
    expect(cropBox(det("person", 1, [10, 20, 110, 220]).box, 640, 480)).toEqual([
      10, 20, 100, 200,
    ]);
  });

  it("clamps a box that runs off the edge rather than wrapping it", () => {
    // A person half out of frame is still a person, and `RawImage.crop` with a
    // negative origin produces a garbage buffer rather than an error.
    expect(cropBox(det("person", 1, [-30, -10, 50, 40]).box, 640, 480)).toEqual([
      0, 0, 50, 40,
    ]);
    expect(
      cropBox(det("person", 1, [600, 440, 900, 700]).box, 640, 480),
    ).toEqual([600, 440, 40, 40]);
  });

  it("rejects a box that clamps to nothing", () => {
    // Feeding the pose model an empty image is worse than dropping the person.
    expect(cropBox(det("person", 1, [700, 500, 900, 700]).box, 640, 480)).toBeNull();
    expect(cropBox(det("person", 1, [10, 10, 10, 10]).box, 640, 480)).toBeNull();
  });
});

describe("toSourceKeypoints", () => {
  it("adds the crop's origin, which the processor never does", () => {
    // `post_process_pose_estimation` scales the heatmap peak by the box's
    // *size* and stops there. Hand-computed: a joint at (5, 10) inside a crop
    // whose top-left is (100, 200) is at (105, 210) in the source.
    const out = toSourceKeypoints(
      [
        [5, 10],
        [0, 0],
      ],
      [0.9, 0.1],
      [0, 16],
      [100, 200, 50, 80],
    );
    expect(out[0]).toEqual({ x: 105, y: 210, score: 0.9, index: 0 });
    expect(out[1]).toEqual({ x: 100, y: 200, score: 0.1, index: 16 });
  });

  it("maps the centre of a crop to the centre of that crop's box", () => {
    // The single assertion that catches the skeleton-floating-beside-the-person
    // bug: a joint at the middle of a 50x80 crop at (100, 200) must land at
    // (125, 240).
    const [centre] = toSourceKeypoints([[25, 40]], [1], [0], [100, 200, 50, 80]);
    expect(centre.x).toBe(125);
    expect(centre.y).toBe(240);
  });

  it("is the identity for a crop at the origin", () => {
    // Which is exactly why the missing offset is invisible in the
    // single-person, whole-image example upstream ships.
    const [only] = toSourceKeypoints([[7, 9]], [1], [0], [0, 0, 100, 100]);
    expect([only.x, only.y]).toEqual([7, 9]);
  });

  it("falls back to the position index when labels are short", () => {
    const out = toSourceKeypoints([[1, 1], [2, 2]], [1], [], [0, 0, 10, 10]);
    expect(out.map((k) => k.index)).toEqual([0, 1]);
    expect(out[1].score).toBe(0);
  });
});

describe("scalePeople", () => {
  const person: Person = {
    box: { xmin: 10, ymin: 20, xmax: 30, ymax: 40 },
    score: 0.9,
    keypoints: [{ x: 15, y: 25, score: 0.8, index: 0 }],
  };

  it("scales the box and the joints together", () => {
    // Scaling one but not the other is a skeleton that shrinks away from its
    // own outline as the source gets larger.
    const [out] = scalePeople([person], 2);
    expect(out.box).toEqual({ xmin: 20, ymin: 40, xmax: 60, ymax: 80 });
    expect(out.keypoints[0]).toMatchObject({ x: 30, y: 50, score: 0.8 });
  });

  it("is a no-op at scale 1, but still copies", () => {
    const out = scalePeople([person], 1);
    expect(out[0]).toEqual(person);
    expect(out).not.toBe(person);
  });
});
