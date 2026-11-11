const avatarStage = document.getElementById('portrait')
const avatarVideo = document.getElementById('interviewer-motion')
const loopStart = 1
avatarVideo.loop = false
let playbackVersion = 0

function updateAvatar() {
  const version = ++playbackVersion
  if (avatarStage.dataset.state !== 'speaking') {
    avatarVideo.hidden = true
    avatarVideo.pause()
    if (avatarVideo.readyState >= 1) avatarVideo.currentTime = 0
    return
  }
  if (avatarVideo.readyState < 1) return
  if (avatarVideo.ended) {
    avatarVideo.currentTime = loopStart
  }
  avatarVideo.play().then(() => {
    if (version === playbackVersion) avatarVideo.hidden = false
  }).catch(() => {
    if (version === playbackVersion) avatarVideo.hidden = true
  })
}

avatarVideo.addEventListener('loadedmetadata', updateAvatar)
avatarVideo.addEventListener('ended', updateAvatar)
new MutationObserver(updateAvatar).observe(avatarStage, {attributes: true, attributeFilter: ['data-state']})
updateAvatar()
