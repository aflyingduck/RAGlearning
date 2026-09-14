# Nimbus Hub — Troubleshooting

**Status ring is solid red.**
This means the Hub lost its Wi-Fi connection. Check that your router is
online. If it is, hold the pairing button for 8 seconds to restart the Hub's
network stack (this does not erase your automations). If the ring stays red
after restart, move the Hub closer to the router and try again.

**Status ring is flashing red.**
This indicates a firmware update failure. Do not unplug the Hub. It will
retry the update automatically up to 3 times over 15 minutes. If it is still
flashing red after that, contact support — do not attempt a factory reset,
as it will not fix a failed firmware update.

**A device won't respond to commands.**
First check the device's own status light. If the Nimbus app shows the
device as "Online" but it doesn't respond, remove and re-pair it from
Settings > Devices > [device name] > Remove, then re-add it. Zigbee devices
more than 10 meters from the Hub (or 5 meters through walls) may also drop
commands intermittently — consider a Zigbee range extender.

**Hub Mesh shows a paired hub as offline.**
Both hubs must be on the same local network and have direct IP connectivity;
Hub Mesh does not work across separate VLANs or guest networks. If hubs are
on different subnets, mesh linking will fail silently rather than showing an
error.

**Factory reset.**
Hold the pairing button for 15 seconds until the status ring flashes purple,
then release. This erases all automations, paired devices, and Wi-Fi
credentials. It does not affect Hub Mesh links on other hubs, which will need
to be removed manually.
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
