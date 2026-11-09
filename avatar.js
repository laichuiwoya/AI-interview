import * as THREE from '/vendor/three.module.min.js'
import {GLTFLoader} from '/vendor/GLTFLoader.js'

const stage = document.getElementById('portrait')
const canvas = document.getElementById('avatar-canvas')
const demo = document.getElementById('avatar-motion')
const inspect = document.getElementById('avatar-demo')
const help = document.getElementById('avatar-help')
const corner = stage.querySelector('.avatar-corner')
const motion = matchMedia('(prefers-reduced-motion: reduce)')

function buildRoom(scene, deskY, chairY) {
  const room = new THREE.Group()
  const surface = (color, roughness = .83) => new THREE.MeshStandardMaterial({color, roughness})
  const wall = surface(0x536968)
  const wood = surface(0x654332, .52)
  const darkWood = surface(0x443023)
  const upholstery = surface(0x243638)
  const trim = surface(0x293a3b)
  const floor = surface(0x787066)
  const glass = new THREE.MeshBasicMaterial({color: 0xa6c1c1})

  function box(material, x, y, z, width, height, depth) {
    const object = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material)
    object.position.set(x, y, z)
    object.receiveShadow = true
    room.add(object)
    return object
  }

  box(floor, 0, -.045, 0, 12, .08, 12)
  box(wall, 0, 1.45, -1.10, 6, 3, .10)
  box(darkWood, 0, .28, -1.04, 6, .56, .06)
  box(trim, -.92, 1.20, -1.025, .67, 1.88, .036)
  box(glass, -.92, 1.20, -1.00, .59, 1.78, .016)
  box(trim, -.92, 1.20, -.988, .025, 1.82, .012)
  box(trim, -.92, 1.10, -.988, .62, .028, .012)
  for (let i = 0; i < 16; i++) box(darkWood, .52 + i * .065, 1.43, -1.03, .028, 2.86, .048)

  box(upholstery, 0, chairY + .24, -.181, .47, .48, .090).rotation.x = -.10
  box(upholstery, 0, chairY, .015, .45, .072, .42)
  box(wood, 0, deskY - .026, .655, 1.65, .052, 1.07)
  box(darkWood, 0, deskY - .061, 1.17, 1.64, .023, .04)
  const cover = box(surface(0x253537), .385, deskY + .006, .463, .148, .012, .113)
  cover.rotation.y = -.18
  box(surface(0xded6c7), .385, deskY + .014, .463, .137, .005, .104).rotation.y = -.18
  box(surface(0x202629), .409, deskY + .019, .461, .006, .005, .078).rotation.y = .35
  scene.add(room)
  return room
}

async function mount() {
  const {scene: character, animations} = await new GLTFLoader().loadAsync('/interviewer.glb')
  const mixer = new THREE.AnimationMixer(character)
  const clips = Object.fromEntries(animations.map(clip => [clip.name.toLowerCase(), mixer.clipAction(clip)]))
  for (const name of ['idle', 'listening', 'speaking', 'thinking']) {
    if (!clips[name]) throw new Error(`Missing ${name} animation`)
  }
  let current = clips.idle.play()
  mixer.update(0)
  character.traverse(object => {
    if (!object.isMesh) return
    object.castShadow = true
    object.receiveShadow = false
    if (object.material.name === 'Eyes') object.material.depthWrite = false
    if (object.material.transparent && object.material.map && object.material.name !== 'Eyes') {
      object.material.alphaTest = .34
      object.material.transparent = false
      object.material.depthWrite = true
      object.material.needsUpdate = true
    }
  })

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x516567)
  scene.add(character)
  const position = name => character.getObjectByName(name).getWorldPosition(new THREE.Vector3())
  const eye = position('eyeL')
  const wrist = position('wristL')
  character.position.y = .075 - position('footL').y
  const eyeY = eye.y + character.position.y
  const deskY = wrist.y + character.position.y - .037
  const chairY = position('upperleg01L').y - .087
  const room = buildRoom(scene, deskY, chairY)

  const skeleton = new THREE.SkeletonHelper(character)
  skeleton.material.vertexColors = false
  skeleton.material.color.set(0x80f3ff)
  skeleton.material.depthTest = false
  skeleton.material.transparent = true
  skeleton.material.opacity = .85
  skeleton.renderOrder = 10
  skeleton.visible = false
  scene.add(skeleton)

  scene.add(new THREE.HemisphereLight(0xf1f6f2, 0x716763, 2.05))
  const key = new THREE.DirectionalLight(0xffe4cc, 2.55)
  key.position.set(-2.1, 3.1, 2.4)
  key.target.position.y = eyeY * .78
  key.castShadow = true
  key.shadow.mapSize.set(1024, 1024)
  key.shadow.camera.left = key.shadow.camera.bottom = -1.4
  key.shadow.camera.right = key.shadow.camera.top = 1.4
  key.shadow.normalBias = .018
  scene.add(key, key.target)
  const rim = new THREE.DirectionalLight(0xb9ddf6, 1.3)
  rim.position.set(1.7, 2.6, -.9)
  scene.add(rim)

  const renderer = new THREE.WebGLRenderer({canvas, antialias: true})
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75))
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  const camera = new THREE.PerspectiveCamera(31, 1, .025, 16)
  let orbit = false
  let angle = .53
  let pointer = null
  let demoStart = null
  let frame = 0
  let previous = 0
  let visible = true
  let lost = false
  let needsDraw = true
  let lastState
  let stateSince = 0

  function cameraPose() {
    if (orbit) {
      camera.position.set(Math.sin(angle) * 2.78, eyeY + .18, Math.cos(angle) * 2.78 + .16)
      camera.lookAt(0, eyeY * .53, .16)
    } else {
      camera.position.set(0, eyeY + .013, 1.66)
      camera.lookAt(0, eyeY - .224, .04)
    }
  }

  function draw(now) {
    frame = 0
    if (lost || !visible || document.hidden) return
    const delta = previous ? Math.min((now - previous) / 1000, .05) : 0
    previous = now
    const elapsed = demoStart === null ? 0 : (now - demoStart) / 1000
    if (elapsed >= 6.6) stopDemo()
    const demoState = elapsed < 2.12 ? 'listening' : elapsed < 4.77 ? 'speaking' : 'thinking'
    const state = demoStart === null ? stage.dataset.state : demoState
    if (state !== lastState) {
      lastState = state
      stateSince = now
    }

    let pose = state
    if (motion.matches || (state === 'listening' && (now - stateSince) / 1000 % 8 > 2.12)) pose = 'idle'
    const next = clips[pose] || clips.idle
    if (next !== current) {
      current.fadeOut(.2)
      current = next.reset().setEffectiveWeight(1).fadeIn(.2).play()
    }
    if (motion.matches) {
      mixer.stopAllAction()
      current = clips.idle.reset().play()
      mixer.update(0)
    } else {
      mixer.update(delta)
    }
    renderer.render(scene, camera)
    needsDraw = false
    if (!motion.matches) queue()
  }

  function queue() {
    if (!frame && !lost && visible && !document.hidden) frame = requestAnimationFrame(draw)
  }

  function invalidate() {
    needsDraw = true
    queue()
  }

  function stopDemo() {
    demoStart = null
    demo.textContent = '演示动作'
    demo.setAttribute('aria-pressed', 'false')
  }

  function toggleInspection() {
    orbit = !orbit
    skeleton.visible = orbit
    room.visible = !orbit
    scene.background.set(orbit ? 0x233338 : 0x516567)
    corner.textContent = orbit ? '3D 人物 / 骨架' : '面试室 / 候选人视角'
    inspect.textContent = orbit ? '返回面试' : '查看骨架'
    inspect.setAttribute('aria-pressed', String(orbit))
    canvas.tabIndex = orbit ? 0 : -1
    canvas.dataset.inspecting = String(orbit)
    help.hidden = !orbit
    if (orbit) angle = .53
    cameraPose()
    invalidate()
  }

  function resize() {
    if (lost) return
    const {clientWidth: width, clientHeight: height} = canvas.parentElement
    camera.aspect = Math.max(width, 1) / Math.max(height, 1)
    camera.updateProjectionMatrix()
    renderer.setSize(Math.max(width, 1), Math.max(height, 1), false)
    invalidate()
  }

  demo.addEventListener('click', () => {
    if (demoStart !== null) stopDemo()
    else {
      demoStart = performance.now()
      lastState = undefined
      demo.textContent = '停止演示'
      demo.setAttribute('aria-pressed', 'true')
    }
    invalidate()
  })
  inspect.addEventListener('click', toggleInspection)
  canvas.addEventListener('pointerdown', event => {
    if (!orbit || !event.isPrimary || event.button !== 0) return
    pointer = {id: event.pointerId, x: event.clientX, angle}
    canvas.setPointerCapture(pointer.id)
  })
  canvas.addEventListener('pointermove', event => {
    if (!orbit || pointer?.id !== event.pointerId) return
    angle = pointer.angle + (event.clientX - pointer.x) * .012
    cameraPose()
    invalidate()
  })
  canvas.addEventListener('lostpointercapture', () => { pointer = null })
  canvas.addEventListener('keydown', event => {
    if (!orbit || !['ArrowLeft', 'ArrowRight', 'Home'].includes(event.key)) return
    event.preventDefault()
    angle = event.key === 'Home' ? .53 : angle + (event.key === 'ArrowLeft' ? -.24 : .24)
    cameraPose()
    invalidate()
  })
  motion.addEventListener('change', () => {
    stopDemo()
    demo.hidden = motion.matches || lost
    invalidate()
  })
  canvas.addEventListener('webglcontextlost', () => {
    lost = true
    stopDemo()
    canvas.hidden = inspect.hidden = demo.hidden = help.hidden = true
  })
  document.addEventListener('visibilitychange', () => {
    if (needsDraw || !motion.matches) queue()
  })

  cameraPose()
  resize()
  if (frame) cancelAnimationFrame(frame)
  draw(performance.now())
  canvas.hidden = inspect.hidden = false
  demo.hidden = motion.matches
  new ResizeObserver(resize).observe(canvas.parentElement)
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting
    if (needsDraw || !motion.matches) queue()
  }).observe(canvas)
}

mount().catch(error => console.warn('Interviewer preview:', error.message))
