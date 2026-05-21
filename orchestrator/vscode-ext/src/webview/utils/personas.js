/**
 * Agent persona mappings -- anthropomorphizes graph nodes with
 * human-like engineering role titles and consistent avatars.
 *
 * Avatars are seeded by persona name so the same "person" always
 * looks the same across all nodes they own.
 *
 * Avatars are cached locally as bundled data URIs so the webview
 * works offline and in restricted environments (VS Code webview,
 * Cursor browser) without hitting the DiceBear CDN.
 */

import AVATAR_CACHE from '../assets/avatar-cache';

const AVATAR_STYLE = 'avataaars';

const ARCHITECTURE_PERSONAS = {
  'Gather Requirements':    { persona: 'Systems Architect (AI)',     avatarStyle: AVATAR_STYLE },
  'System Architecture':    { persona: 'Silicon Engineer (AI)',       avatarStyle: AVATAR_STYLE },
  'Functional Requirements':{ persona: 'Performance Engineer (AI)',  avatarStyle: AVATAR_STYLE },
  'Block Diagram':          { persona: 'Systems Architect (AI)',     avatarStyle: AVATAR_STYLE },
  'Memory Map':             { persona: 'Memory Engineer (AI)',       avatarStyle: AVATAR_STYLE },
  'Clock Tree':             { persona: 'Clock Engineer (AI)',        avatarStyle: AVATAR_STYLE },
  'Register Spec':          { persona: 'Register Lead (AI)',         avatarStyle: AVATAR_STYLE },
  'Constraint Check':       { persona: 'Design Reviewer (AI)',       avatarStyle: AVATAR_STYLE },
  'Finalize Architecture':  { persona: 'Design Reviewer (AI)',       avatarStyle: AVATAR_STYLE },
  'Create Documentation':   { persona: 'Tech Writer (AI)',           avatarStyle: AVATAR_STYLE },
  'Architecture Complete':  { persona: 'Design Reviewer (AI)',       avatarStyle: AVATAR_STYLE },
  'Constraint Iteration':   { persona: 'Design Reviewer (AI)',       avatarStyle: AVATAR_STYLE },
  'Abort':                  { persona: 'You',                        avatarStyle: 'adventurer-neutral' },
  'Escalate PRD':           { persona: 'You',                        avatarStyle: 'adventurer-neutral' },
  'Escalate Diagram':       { persona: 'You',                        avatarStyle: 'adventurer-neutral' },
  'Escalate Constraints':   { persona: 'You',                        avatarStyle: 'adventurer-neutral' },
  'Escalate Exhausted':     { persona: 'You',                        avatarStyle: 'adventurer-neutral' },
  'Final Review':           { persona: 'You',                        avatarStyle: 'adventurer-neutral' },
};

const FRONTEND_PERSONAS = {
  'init_block':            { persona: 'Build System (AI)',            avatarStyle: AVATAR_STYLE },
  'generate_uarch_spec':   { persona: 'RTL Architect (AI)',           avatarStyle: AVATAR_STYLE },
  'review_uarch_spec':     { persona: 'You',                         avatarStyle: 'adventurer-neutral' },
  'generate_rtl':          { persona: 'RTL Engineer (AI)',            avatarStyle: AVATAR_STYLE },
  'lint':                  { persona: 'QA Engineer (EDA)',            avatarStyle: AVATAR_STYLE },
  'generate_testbench':    { persona: 'Verification Engineer (AI)',   avatarStyle: AVATAR_STYLE },
  'simulate':              { persona: 'Verification Engineer (EDA)',  avatarStyle: AVATAR_STYLE },
  'synthesize':            { persona: 'PD Lead (AI)',                 avatarStyle: AVATAR_STYLE },
  'diagnose':              { persona: 'Design Lead (AI)',             avatarStyle: AVATAR_STYLE },
  'decide':                { persona: 'Design Lead (AI)',             avatarStyle: AVATAR_STYLE },
  'ask_human':             { persona: 'You',                         avatarStyle: 'adventurer-neutral' },
  'increment_attempt':     { persona: 'Build System (AI)',            avatarStyle: AVATAR_STYLE },
  'block_done':            { persona: 'Build System (AI)',            avatarStyle: AVATAR_STYLE },
  'init_tier':             { persona: 'Build System (AI)',            avatarStyle: AVATAR_STYLE },
  'process_block':         { persona: 'Build System (AI)',            avatarStyle: AVATAR_STYLE },
  'advance_tier':          { persona: 'Build System (AI)',            avatarStyle: AVATAR_STYLE },
  'pipeline_complete':     { persona: 'Build System (AI)',            avatarStyle: AVATAR_STYLE },
  'integration_check':     { persona: 'Integration Engineer (AI)',    avatarStyle: AVATAR_STYLE },
  'integration_dv':        { persona: 'Lead DV Engineer (AI)',        avatarStyle: AVATAR_STYLE },
};

const BACKEND_PERSONAS = {
  'init_design':           { persona: 'Build System (AI)',        avatarStyle: AVATAR_STYLE },
  'flat_top_synthesis':    { persona: 'Synthesis Engineer (AI)',   avatarStyle: AVATAR_STYLE },
  'floorplan':             { persona: 'PD Engineer (AI)',         avatarStyle: AVATAR_STYLE },
  'place':                 { persona: 'PD Engineer (AI)',         avatarStyle: AVATAR_STYLE },
  'cts':                   { persona: 'Clock Engineer (AI)',      avatarStyle: AVATAR_STYLE },
  'route':                 { persona: 'PD Engineer (AI)',         avatarStyle: AVATAR_STYLE },
  'run_pnr':               { persona: 'PD Engineer (AI)',         avatarStyle: AVATAR_STYLE },
  'drc':                   { persona: 'DRC Engineer (AI)',        avatarStyle: AVATAR_STYLE },
  'lvs':                   { persona: 'LVS Engineer (AI)',        avatarStyle: AVATAR_STYLE },
  'timing_signoff':        { persona: 'Timing Engineer (AI)',     avatarStyle: AVATAR_STYLE },
  'mpw_precheck':          { persona: 'Tapeout Engineer (AI)',    avatarStyle: AVATAR_STYLE },
  'power_analysis':        { persona: 'Power Engineer (AI)',      avatarStyle: AVATAR_STYLE },
  'diagnose':              { persona: 'PD Lead (AI)',             avatarStyle: AVATAR_STYLE },
  'decide':                { persona: 'PD Lead (AI)',             avatarStyle: AVATAR_STYLE },
  'ask_human':             { persona: 'You',                     avatarStyle: 'adventurer-neutral' },
  'increment_attempt':     { persona: 'Build System (AI)',        avatarStyle: AVATAR_STYLE },
  'advance_block':         { persona: 'Build System (AI)',        avatarStyle: AVATAR_STYLE },
  'backend_complete':      { persona: 'Build System (AI)',        avatarStyle: AVATAR_STYLE },
  'generate_3d_view':      { persona: 'PD Engineer (AI)',         avatarStyle: AVATAR_STYLE },
  'final_report':          { persona: 'Tech Writer (AI)',         avatarStyle: AVATAR_STYLE },
  'generate_wrapper':      { persona: 'PD Engineer (AI)',         avatarStyle: AVATAR_STYLE },
  'wrapper_pnr':           { persona: 'PD Engineer (AI)',         avatarStyle: AVATAR_STYLE },
  'wrapper_drc':           { persona: 'DRC Engineer (AI)',        avatarStyle: AVATAR_STYLE },
  'wrapper_lvs':           { persona: 'LVS Engineer (AI)',        avatarStyle: AVATAR_STYLE },
  'tapeout_complete':      { persona: 'Tapeout Engineer (AI)',    avatarStyle: AVATAR_STYLE },
};

const GRAPH_PERSONAS = {
  architecture: ARCHITECTURE_PERSONAS,
  frontend:     FRONTEND_PERSONAS,
  backend:      BACKEND_PERSONAS,
};

/**
 * Look up the persona for a given node.
 * @param {string} graphName  - 'architecture' | 'frontend' | 'backend'
 * @param {string} nodeId     - The node ID from the graph
 * @returns {{ persona: string, avatarStyle: string } | null}
 */
export function getPersona(graphName, nodeId) {
  const map = GRAPH_PERSONAS[graphName];
  return map?.[nodeId] ?? null;
}

const PROFESSIONAL_TOPS = [
  'bob', 'bun', 'curly', 'curvy', 'longButNotTooLong', 'miaWallace',
  'straight01', 'straight02', 'straightAndStrand',
  'shortCurly', 'shortFlat', 'shortRound', 'shortWaved',
  'sides', 'theCaesar', 'theCaesarAndSidePart',
].join(',');

const ALLOWED_MOUTHS = [
  'concerned', 'default', 'disbelief', 'eating', 'grimace',
  'serious', 'smile', 'twinkle',
].join(',');

/**
 * Build an avatar URL for a persona. Returns a locally cached data URI
 * when available, falling back to the DiceBear CDN for unknown personas.
 */
export function personaAvatarUrl(persona, style = 'avataaars', size = 36) {
  const cached = AVATAR_CACHE[persona];
  if (cached) return cached;

  const base =
    `https://api.dicebear.com/9.x/${style}/svg` +
    `?seed=${encodeURIComponent(persona)}` +
    `&size=${size}` +
    `&backgroundColor=transparent`;

  if (style === 'avataaars') {
    return (
      base +
      `&top=${PROFESSIONAL_TOPS}` +
      `&mouth=${ALLOWED_MOUTHS}` +
      `&clothing=collarAndSweater` +
      `&skinColor=ae5d29,d08b5b,edb98a,ffdbb4` +
      `&accessories=kurt,prescription01,prescription02,round,sunglasses,wayfarers`
    );
  }

  return base;
}
