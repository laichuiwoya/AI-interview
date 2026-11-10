const avatarStage = document.getElementById('portrait')
const avatarVideo = document.getElementById('interviewer-motion')
let playbackVersion = 0

function updateAvatar() {
  const version = ++playbackVersion
  if (avatarStage.dataset.state !== 'speaking') {
    avatarVideo.hidden = true
    avatarVideo.pause()
    avatarVideo.currentTime = 0
    return
  }
  avatarVideo.play().then(() => {
    if (version === playbackVersion) avatarVideo.hidden = false
  }).catch(() => {
    if (version === playbackVersion) avatarVideo.hidden = true
  })
}

new MutationObserver(updateAvatar).observe(avatarStage, {attributes: true, attributeFilter: ['data-state']})
updateAvatar()
