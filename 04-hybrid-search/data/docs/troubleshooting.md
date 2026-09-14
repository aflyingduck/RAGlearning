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

**Error codes.**
If the app shows a specific error code, look it up here rather than
guessing from the status ring color alone:
- `ERR-4471` — firmware signature verification failed during an update.
  Not the same as a normal flashing-red retry; this one won't resolve on
  its own. Contact support with this code.
- `ERR-2208` — Zigbee radio failed to initialize at boot. Usually caused by
  a damaged antenna connector; not fixable via factory reset.
- `ERR-1090` — Hub Mesh link rejected because the two hubs are running
  different firmware versions. Update both hubs to the same version before
  retrying the mesh link.

**Factory reset.**
Hold the pairing button for 15 seconds until the status ring flashes purple,
then release. This erases all automations, paired devices, and Wi-Fi
credentials. It does not affect Hub Mesh links on other hubs, which will need
to be removed manually.
