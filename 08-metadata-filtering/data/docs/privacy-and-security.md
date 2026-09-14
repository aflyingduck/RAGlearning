# Nimbus Hub — Privacy and Security

Voice commands are processed locally on the Hub by default using its
on-device voice model. Audio is only sent to Solstice Labs servers if you
explicitly enable "Enhanced Voice Recognition" in Settings > Privacy, which
uses cloud processing for improved accuracy in noisy environments.

When Enhanced Voice Recognition is on, audio clips are retained for 30 days
to improve the model, then deleted. Clips are never used to train models
shared across other customers' accounts. You can delete stored clips
immediately from Settings > Privacy > Manage Voice Data.

The Hub encrypts all local network traffic between itself and paired devices
using per-device keys established during pairing. Hub-to-cloud traffic (for
Nimbus Plus features) uses TLS 1.3.

Camera and sensor data (for compatible third-party devices) is stored
locally on the Hub's onboard storage by default and is not uploaded to the
cloud unless you enable Nimbus Plus remote access, in which case only the
specific stream you are actively viewing remotely is relayed, not recorded
centrally.

Solstice Labs does not sell customer data to third parties. A full data
export can be requested from Settings > Privacy > Export My Data, delivered
within 14 days.
