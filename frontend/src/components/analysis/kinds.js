// What each data type accepts, what its result is called, and the words for the backend's check and attack codes.
// Limits are the benchmark's (docs/*-benchmark-findings.md); update them with the docs when a detector changes.

export const DATA_KINDS = {
  csv: {
    label: 'Table', accept: '.csv', maxMb: 200, noun: 'rows',
    format: 'A CSV with a label column. Text columns with up to 50 distinct values are checked as categories; answer-key columns such as is_poisoned are ignored.',
    limits: ['Below about 3% poison the batch verdict is weak; the ranked rows are still worth reading.',
      'Rows moved towards a victim row with correct labels (feature collision) are found as rows, rarely as cells.'],
  },
  images: {
    label: 'Images', accept: '.zip', maxMb: 500, noun: 'images',
    format: 'A zip with one folder per class (cat/001.png). Up to 6,000 images are used, resized to 32×32.',
    limits: ['Images are checked at 32×32: a trigger smaller than about 1/32 of the image can vanish.',
      'At 1% poison, label flips are ranked but rarely give a batch verdict (about 1 in 10 batches).'],
  },
  audio: {
    label: 'Audio', accept: '.zip', maxMb: 500, noun: 'clips',
    format: 'A zip of PCM WAV files with one folder per class (dog/001.wav). Clips are mixed to mono and padded or trimmed to one length.',
    limits: ['Triggers added to correctly labelled clips (clean-label) are not detected.',
      'Small sets (a few hundred clips) rarely give a batch verdict at 1% poison.'],
  },
  video: {
    label: 'Video', accept: '.zip', maxMb: 500, noun: 'clips',
    format: 'A zip of video files (mp4, avi, mov, mkv, webm) with one folder per class. 8 frames per clip are checked.',
    limits: ['Only triggers present on every frame were benchmarked; a moving or brief trigger can blur out.',
      'Triggers added to correctly labelled clips (clean-label) are not detected.'],
  },
};

/** History sources that are dataset checks, by data type. Model scans are not here. */
export const KIND_OF_SOURCE = { upload: 'csv', real_dataset: 'csv', demo: 'csv', image_upload: 'images', audio_upload: 'audio', video_upload: 'video' };

/** The data type of a result (a table result may carry no source when it comes straight from an upload). */
export function kindOf(result) {
  return KIND_OF_SOURCE[result?.source] || 'csv';
}

// [name, what it looks for]. Table checks are label_check / shift_check / resid_check; media checks are short codes.
export const CHECK_NAMES = {
  label_check: ['Label check', 'labels that do not fit the rest of the row'],
  shift_check: ['Distribution-shift check', 'incoming rows a model can tell apart from the reference'],
  resid_check: ['Column-consistency check', 'rows whose columns no longer fit together'],
  label: ['Labels', 'labels that do not fit the content'],
  shift: ['Content shift', 'incoming items that look unlike the reference'],
  pixel: ['Pixel statistics', 'patches, noise and blending'],
  dup: ['Near-duplicates', 'copies of the same item'],
  cell: ['Shared patch', 'one local pattern repeated across items'],
  resid: ['Shared faint pattern', 'a faint pattern repeated across items'],
  chroma: ['Colour noise', 'noise in flat regions'],
  sig: ['Within-class pattern', 'a pattern shared by part of one class, labels left correct'],
};
export const checkName = (key) => (CHECK_NAMES[key] || [String(key).replace(/_/g, ' ')])[0];

export const ATTACK_NAMES = {
  label_flip_random: 'Random label flipping',
  label_flip_targeted: 'Targeted label flipping',
  backdoor_ood: 'Backdoor trigger, out-of-range values',
  backdoor_inrange: 'Backdoor trigger, in-range values',
  feature_collision: 'Feature collision (clean-label)',
  covariate_shift: 'Covariate shift or corrupted source',
};

export const PROFILE_FEATURES = {
  label_evidence: 'Label check evidence',
  shift_evidence: 'Shift check evidence',
  resid_evidence: 'Column-consistency evidence',
  label_blamed: 'Share of rows whose label is the top cell',
  trigger_columns: 'Share whose top cell is in the 3 commonest columns',
  label_concentration: 'Over-representation of their commonest label',
  out_of_range: 'Share with a value outside the reference range',
  tightness: 'Spread of the rows relative to the reference',
  unusual_columns: 'Share of their columns that look unusual',
};
