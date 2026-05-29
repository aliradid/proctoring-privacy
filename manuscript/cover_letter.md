# Cover letter, submission to *Informatica* (Slovenia)

Radid Ali, Ghazouani Mohamed and El habib Benlahmar
Department of Mathematics and Computer Science
Hassan II University, Faculty of Sciences Ben M'sik
Casablanca, Morocco
*radidalix@outlook.fr*

21 May 2026

To the Editor-in-Chief and the Editorial Board of *Informatica*

Dear Editors,

We are pleased to submit our manuscript entitled **"A Privacy-Preserving Multi-Modal Proctoring Framework Using Deep Learning for Secure Online Examinations"** for consideration as a research article in *Informatica*.

Online examination platforms have become a central part of higher education in the post-COVID era, and the proctoring systems that police them are increasingly dependent on biometric identification such as continuous facial recognition, voiceprint extraction, and speech transcription. These methods conflict directly with the European General Data Protection Regulation (GDPR Article 9) and, in our jurisdiction, with the Moroccan CNDP Law 09-08, both of which classify biometric data as a special category requiring strict justification. The result is a deployment gap: institutions need scalable remote assessment but cannot legally adopt the dominant commercial solutions.

Our paper closes this gap with a privacy-by-design multi-modal architecture that integrates YOLOv8-s for contextual object detection, MediaPipe FaceMesh for non-identifying geometric behavioural cues, and a Whisper-Base encoder operating in content-free mode for acoustic anomaly detection. The three modality outputs are converted into a standardised metadata event stream and combined through a learned Suspicion Index Fusion Model. The system never stores biometric templates, raw video, or raw audio.

We believe the manuscript is a good fit for *Informatica* for four reasons:

1. **Scope.** The paper is squarely within "intelligent systems and software" (multi-modal deep learning, real-time inference, statistical decision fusion) and directly addresses the privacy and ethical dimensions of modern AI deployment in education.
2. **Real-data evaluation.** Every benchmark reported uses publicly licensed real data: 217 COCO val2017 images for the visual subsystem, 400 LibriSpeech clips for the acoustic subsystem, 400 trajectories built from real MediaPipe FaceMesh landmarks for the behavioural subsystem, and 300 fusion-evaluation sessions whose feature vectors are real outputs of the three modality encoders.
3. **Statistical rigour.** Five fusion strategies are compared under 5-fold stratified cross-validation with bootstrap confidence intervals and pairwise McNemar tests. The random-forest fusion (F1 = 0.905 ± 0.041, 95% CI [0.876, 0.931]) matches a transparent weighted-sum baseline and significantly outperforms the weaker learned variants; we report the non-significant comparisons openly rather than overstating the result.
4. **Adversarial privacy verification.** The most novel contribution is a quantitative privacy test on 30 real LibriSpeech speakers: a raw-audio attacker achieves a re-identification AUC of 0.994, while the same attacker constrained to the metadata stream emitted by our system achieves only 0.478 (chance level). To the best of our knowledge no prior proctoring paper provides such empirical evidence.

The full source code, configuration files, and bootstrap scripts are released under the MIT license at `https://github.com/aliradid/proctoring-privacy` and the experiments can be reproduced end-to-end with a single command on a consumer workstation.

The manuscript has not been published elsewhere and is not currently under consideration by any other journal. All authors have read and approved the submission. There is no conflict of interest to declare, and we do not have any funding source to report for this work. We additionally note, in line with the journal's policy, that limited use was made of language-support tools for grammar and clarity editing of the English prose; the scientific content and all conclusions are the authors' own.

We would like to suggest the following potential reviewers, none of whom has a conflict of interest with the authors:

- Prof. Joon Son Chung (KAIST): speech/audio overlap detection and multi-modal deep learning
- Prof. Karen Renaud (University of Strathclyde): usable security and authentication
- Prof. Enkelejda Kasneci (Technical University of Munich): privacy-preserving gaze and eye-tracking
- Prof. Mohammed M. Masud (UAE University): automated proctoring and academic integrity

Thank you for considering our submission. We look forward to the editorial board's evaluation and are happy to provide any additional material on request.

Sincerely,

Radid Ali, on behalf of all authors
