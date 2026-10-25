# Engineering Practices

## Code review

Every change to the flight-control firmware needs approval from two reviewers, at least one of whom must be on the Flight Safety team. Other repositories need one approving review.
Pull requests should stay under 400 changed lines. Larger changes must be split, unless they are generated code.

## Releases

Firmware is released quarterly, in the first week of January, April, July and October. Each release goes through a 3-week beta with at least 15 customer pilot sites before general availability.
Mission Studio, the desktop app, ships every two weeks on Tuesdays.
Emergency hotfixes can skip the beta but require sign-off from the VP of Engineering, Ravi Deshpande.

## On-call

Engineers on the platform team rotate on-call weekly, handing over on Monday at 10:00 IST. On-call engineers receive an allowance of 4,000 INR per week and must acknowledge pages within 15 minutes.

## Incident severity levels

- SEV1: drones grounded or a safety risk for customers. All hands; status page updated every 30 minutes.
- SEV2: a major feature is unavailable for many customers, with no safety impact. Update every 2 hours.
- SEV3: minor issue or a workaround exists. Fixed during business hours.
A written post-mortem is required within 5 business days for every SEV1 and SEV2 incident. Post-mortems are blameless.

## Tech stack

Flight-control firmware is written in Rust and runs on a real-time operating system. Mission Studio is built with TypeScript and Electron. Backend services are written in Go and run on Kubernetes in the AWS ap-south-1 (Mumbai) region.
