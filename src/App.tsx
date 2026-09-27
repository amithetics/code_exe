import { useMemo, useState, useRef, useEffect, type ReactNode, type DragEvent, type ChangeEvent } from 'react';
import {
  ArrowDownToLine, ArrowUpRight, BarChart3, BookOpen, Check, ChevronDown,
  CircleHelp, Database, FileCheck2, FlaskConical, Gauge, Menu, Network,
  Play, Search, Settings, ShieldCheck, Sparkles, Upload, X, Zap,
  RotateCcw, FileText, AlertCircle, Eye, Archive, Sliders, ChevronLeft, ChevronRight
} from 'lucide-react';

type View = 'overview' | 'datasets' | 'experiments' | 'submissions' | 'leaderboard';
type RecordRow = { id: string; name: string; address: string; country: string; source: 'S1' | 'S2' | 'S3'; matchId?: string };
type Run = {
  id: string;
  name: string;
  score: number | null;
  precision: number | null;
  recall: number | null;
  candidates: number;
  matchesCount: number;
  threshold: number;
  nameWeight: number;
  blockingStrategy: BlockingStrategy;
  created: string;
  status: 'Complete' | 'Running';
};
type UploadedFileMeta = { name: string; rows: number; source: 'S1' | 'S2' | 'S3' };
type BlockingStrategy = 'country' | 'country_prefix' | 'prefix_3' | 'exhaustive';
type SimilarityMetric = 'hybrid' | 'token_jaccard' | 'levenshtein';

type CandidateMatch = {
  candidate: RecordRow;
  score: number;
  nameScore: number;
  addrScore: number;
  isMatch: boolean;
};

const baseEntities = [
  ['Northstar Analytics', '1200 Market Street, San Francisco, CA', 'US'],
  ['Bharat Foods Pvt Ltd', '14 MG Road, Bengaluru, Karnataka', 'IN'],
  ['Maple & Co. Logistics', '88 Main Street, Chicago, IL', 'US'],
  ['Horizon Medical Group', '44 Park Avenue, New York, NY', 'US'],
  ['Lumen Retail Limited', '7 Oxford Road, Austin, TX', 'US'],
  ['Saffron Cloud Services', '2 Cyber City, Hyderabad, Telangana', 'IN'],
  ['Alpine Outdoor Supply', '19 Trail Road, Denver, CO', 'US'],
  ['Bluebird Creative Studio', '31 Collins Street, Boston, MA', 'US'],
  ['Cedar Ridge Capital', '560 Pine Street, Seattle, WA', 'US'],
  ['Vela Manufacturing', '9 Industrial Estate, Pune, Maharashtra', 'IN'],
  ['Pioneer Education Trust', '12 College Road, Boston, MA', 'US'],
  ['Amber Home Goods', '63 Market Road, Mumbai, Maharashtra', 'IN'],
];

const source2Edits = [
  'Northstar Analytics Inc.', 'Bharat Foods Pvt. Ltd.', 'Maple Co Logistics',
  'Horizon Med Group', 'Lumen Retail Ltd', 'Saffron Cloud Svcs',
  'Alpine Outdoor Supply AG', 'Bluebird Creative', 'Cedar Ridge Cap.',
  'Vela Mfg Pvt Ltd', 'Pioneer Education', 'Amber Home Goods SAS'
];

const source3Edits = [
  'North Star Analytics', 'Bharat Foods', 'Maple & Company Logistics',
  'Horizon Medical', 'Lumen Retail', 'Saffron Cloud Services Pvt Ltd',
  'Alpine Outdoor', 'Bluebird Creative Studio Pty', 'Cedar Ridge Capital LLC',
  'Vela Manufacturing India', 'Pioneer Education Trust Inc', 'Amber Home Goods'
];

const syntheticRecords: RecordRow[] = baseEntities.flatMap(([name, address, country], index) => [
  { id: `S1-${String(index + 1).padStart(4, '0')}`, name, address, country, source: 'S1' as const },
  { id: `S2-${String(index + 1).padStart(4, '0')}`, name: source2Edits[index], address: address.replace(/\b(Street|Road|Avenue)\b/, (word) => word === 'Street' ? 'St' : word === 'Road' ? 'Rd' : 'Ave'), country, source: 'S2' as const, matchId: `S1-${String(index + 1).padStart(4, '0')}` },
  { id: `S3-${String(index + 1).padStart(4, '0')}`, name: source3Edits[index], address: address.replace(/\b\d+\b/, (value) => String(Number(value) + (index % 3))), country, source: 'S3' as const, matchId: `S1-${String(index + 1).padStart(4, '0')}` },
]);

const navItems: { label: string; view: View; icon: typeof Gauge }[] = [
  { label: 'Overview', view: 'overview', icon: Gauge },
  { label: 'Datasets', view: 'datasets', icon: Database },
  { label: 'Experiments', view: 'experiments', icon: FlaskConical },
  { label: 'Submissions', view: 'submissions', icon: FileCheck2 },
  { label: 'Leaderboard', view: 'leaderboard', icon: BarChart3 },
];

// String Normalization
const cleanText = (str: string) => str.toLowerCase().replace(/[^a-z0-9]/g, '');

// Token Jaccard Similarity
function tokenJaccard(a: string, b: string): number {
  const tokensA = new Set(a.toLowerCase().split(/\s+/).filter(Boolean));
  const tokensB = new Set(b.toLowerCase().split(/\s+/).filter(Boolean));
  if (!tokensA.size && !tokensB.size) return 1;
  if (!tokensA.size || !tokensB.size) return 0;
  let intersection = 0;
  for (const t of tokensA) {
    if (tokensB.has(t)) intersection++;
  }
  return intersection / (tokensA.size + tokensB.size - intersection);
}

// Levenshtein Similarity
function levenshteinSimilarity(a: string, b: string): number {
  const s1 = cleanText(a);
  const s2 = cleanText(b);
  if (!s1.length && !s2.length) return 1;
  if (!s1.length || !s2.length) return 0;
  const maxLen = Math.max(s1.length, s2.length);

  const d: number[][] = Array.from({ length: s1.length + 1 }, () => new Array(s2.length + 1).fill(0));
  for (let i = 0; i <= s1.length; i++) d[i][0] = i;
  for (let j = 0; j <= s2.length; j++) d[0][j] = j;

  for (let i = 1; i <= s1.length; i++) {
    for (let j = 1; j <= s2.length; j++) {
      const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
    }
  }
  return (maxLen - d[s1.length][s2.length]) / maxLen;
}

// Hybrid similarity (Character overlap + Token Jaccard)
function calculateSimilarity(left: string, right: string, metric: SimilarityMetric): number {
  if (metric === 'token_jaccard') return tokenJaccard(left, right);
  if (metric === 'levenshtein') return levenshteinSimilarity(left, right);

  // Hybrid metric (default)
  const a = cleanText(left);
  const b = cleanText(right);
  const longer = Math.max(a.length, b.length);
  if (!longer) return 1;
  let sameChar = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
    if (a[i] === b[i]) sameChar += 1;
  }
  const charOverlap = sameChar / longer;
  const wordOverlap = tokenJaccard(left, right);
  return charOverlap * 0.5 + wordOverlap * 0.5;
}

// Delimited Text Parser
function parseDelimitedText(text: string, filename: string): { rows: RecordRow[]; source: 'S1' | 'S2' | 'S3' } {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) return { rows: [], source: 'S1' };

  const firstLine = lines[0];
  const tabCount = (firstLine.match(/\t/g) || []).length;
  const commaCount = (firstLine.match(/,/g) || []).length;
  const delimiter = tabCount >= commaCount ? '\t' : ',';

  const parseLine = (line: string): string[] => {
    const tokens: string[] = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (ch === delimiter && !inQuotes) {
        tokens.push(cur.trim());
        cur = '';
      } else {
        cur += ch;
      }
    }
    tokens.push(cur.trim());
    return tokens;
  };

  const headers = parseLine(lines[0]).map((h) => h.toLowerCase().replace(/[^a-z0-9]/g, ''));
  const idIdx = headers.findIndex((h) => h === 'id' || h.endsWith('id') || h === 'entityid' || h === 'recordid');
  const nameIdx = headers.findIndex((h) => h.includes('name') || h === 'title' || h === 'company');
  const addrIdx = headers.findIndex((h) => h.includes('address') || h.includes('addr') || h.includes('street') || h === 'location');
  const countryIdx = headers.findIndex((h) => h.includes('country') || h === 'nation' || h === 'state' || h === 'cc');
  const srcIdx = headers.findIndex((h) => h === 'source' || h === 'src' || h === 'dataset');
  const matchIdx = headers.findIndex((h) => h.includes('match') || h.includes('truth') || h.includes('target') || h.includes('groundtruth'));

  const lowerName = filename.toLowerCase();
  let defaultSource: 'S1' | 'S2' | 'S3' = 'S1';
  if (lowerName.includes('source2') || lowerName.includes('s2')) defaultSource = 'S2';
  else if (lowerName.includes('source3') || lowerName.includes('s3')) defaultSource = 'S3';
  else if (lowerName.includes('source1') || lowerName.includes('s1')) defaultSource = 'S1';

  const rows: RecordRow[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = parseLine(lines[i]);
    if (!cols || cols.length === 0 || cols.every((c) => !c)) continue;

    const rowId = idIdx >= 0 && cols[idIdx] ? cols[idIdx] : `R-${String(i).padStart(4, '0')}`;
    const name = nameIdx >= 0 && cols[nameIdx] ? cols[nameIdx] : (cols[0] || 'Unknown');
    const address = addrIdx >= 0 && cols[addrIdx] ? cols[addrIdx] : (cols[1] || '');
    const country = countryIdx >= 0 && cols[countryIdx] ? cols[countryIdx] : (cols[2] || 'GLOBAL');

    let rowSource = defaultSource;
    if (srcIdx >= 0 && cols[srcIdx]) {
      const s = cols[srcIdx].toUpperCase();
      if (s === 'S1' || s === 'S2' || s === 'S3') rowSource = s as 'S1' | 'S2' | 'S3';
    } else if (rowId.startsWith('S1-')) rowSource = 'S1';
    else if (rowId.startsWith('S2-')) rowSource = 'S2';
    else if (rowId.startsWith('S3-')) rowSource = 'S3';

    const matchId = matchIdx >= 0 && cols[matchIdx] ? cols[matchIdx] : undefined;

    rows.push({
      id: rowId,
      name,
      address,
      country,
      source: rowSource,
      matchId,
    });
  }

  return { rows, source: defaultSource };
}

// Dedicated Ground Truth TSV Parser
function parseGroundTruthText(text: string): Map<string, string> {
  const map = new Map<string, string>();
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  for (const line of lines) {
    const parts = line.split(/[\t,]/).map((p) => p.trim());
    if (parts.length >= 2) {
      const s1 = parts[0];
      const matched = parts[1];
      if (s1.toLowerCase().includes('id') && matched.toLowerCase().includes('id')) continue;
      map.set(s1, matched);
    }
  }
  return map;
}

// Native Client-side ZIP Creator (Zero External Dependencies)
function createZip(files: { name: string; content: string }[]): Blob {
  const crcTable = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    }
    crcTable[i] = c;
  }
  const calcCrc = (bytes: Uint8Array) => {
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < bytes.length; i++) {
      crc = (crc >>> 8) ^ crcTable[(crc ^ bytes[i]) & 0xFF];
    }
    return (crc ^ 0xFFFFFFFF) >>> 0;
  };

  const parts: Uint8Array[] = [];
  const centralDirs: Uint8Array[] = [];
  let offset = 0;

  for (const file of files) {
    const encoder = new TextEncoder();
    const fileData = encoder.encode(file.content);
    const fileNameBytes = encoder.encode(file.name);
    const crc = calcCrc(fileData);

    const localHeader = new Uint8Array(30 + fileNameBytes.length);
    const lv = new DataView(localHeader.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, 20, true);
    lv.setUint16(6, 0, true);
    lv.setUint16(8, 0, true);
    lv.setUint16(10, 0, true);
    lv.setUint16(12, 0, true);
    lv.setUint32(14, crc, true);
    lv.setUint32(18, fileData.length, true);
    lv.setUint32(22, fileData.length, true);
    lv.setUint16(26, fileNameBytes.length, true);
    lv.setUint16(28, 0, true);
    localHeader.set(fileNameBytes, 30);

    parts.push(localHeader, fileData);

    const centralHeader = new Uint8Array(46 + fileNameBytes.length);
    const cv = new DataView(centralHeader.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true);
    cv.setUint16(6, 20, true);
    cv.setUint16(8, 0, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, 0, true);
    cv.setUint16(14, 0, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, fileData.length, true);
    cv.setUint32(24, fileData.length, true);
    cv.setUint16(28, fileNameBytes.length, true);
    cv.setUint16(30, 0, true);
    cv.setUint16(32, 0, true);
    cv.setUint16(34, 0, true);
    cv.setUint16(36, 0, true);
    cv.setUint32(38, 0, true);
    cv.setUint32(42, offset, true);
    centralHeader.set(fileNameBytes, 46);
    centralDirs.push(centralHeader);

    offset += localHeader.length + fileData.length;
  }

  const centralOffset = offset;
  let centralSize = 0;
  for (const cd of centralDirs) centralSize += cd.length;

  const endRecord = new Uint8Array(22);
  const ev = new DataView(endRecord.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(4, 0, true);
  ev.setUint16(6, 0, true);
  ev.setUint16(8, files.length, true);
  ev.setUint16(10, files.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, centralOffset, true);
  ev.setUint16(20, 0, true);

  return new Blob([...parts, ...centralDirs, endRecord] as BlobPart[], { type: 'application/zip' });
}

// Entity Matching Engine
function runMatching(
  allRecords: RecordRow[],
  threshold: number,
  nameWeight: number,
  blockingStrategy: BlockingStrategy,
  similarityMetric: SimilarityMetric
) {
  let source1 = allRecords.filter((row) => row.source === 'S1');
  let candidates = allRecords.filter((row) => row.source !== 'S1');

  if (source1.length === 0 && allRecords.length > 0) {
    const splitPoint = Math.max(1, Math.floor(allRecords.length / 3));
    source1 = allRecords.slice(0, splitPoint).map((r) => ({ ...r, source: 'S1' as const }));
    candidates = allRecords.slice(splitPoint).map((r) => ({ ...r, source: 'S2' as const }));
  }

  const addrWeight = Math.max(0, 1 - nameWeight);

  // Score distribution buckets: [0-0.2], [0.2-0.4], [0.4-0.6], [0.6-0.8], [0.8-1.0]
  const histogram = [0, 0, 0, 0, 0];

  const output = source1.map((source) => {
    let pool = candidates;

    if (blockingStrategy === 'country') {
      const match = candidates.filter((c) => c.country.toUpperCase() === source.country.toUpperCase() || c.country === 'GLOBAL');
      if (match.length > 0) pool = match;
    } else if (blockingStrategy === 'country_prefix') {
      const match = candidates.filter(
        (c) =>
          (c.country.toUpperCase() === source.country.toUpperCase() || c.country === 'GLOBAL') &&
          c.name.charAt(0).toUpperCase() === source.name.charAt(0).toUpperCase()
      );
      if (match.length > 0) pool = match;
    } else if (blockingStrategy === 'prefix_3') {
      const p3 = source.name.slice(0, 3).toLowerCase();
      const match = candidates.filter((c) => c.name.slice(0, 3).toLowerCase() === p3);
      if (match.length > 0) pool = match;
    }

    const scoredPool: CandidateMatch[] = pool.map((candidate) => {
      const nameScore = calculateSimilarity(source.name, candidate.name, similarityMetric);
      const addrScore = calculateSimilarity(source.address, candidate.address, similarityMetric);
      const score = nameScore * nameWeight + addrScore * addrWeight;

      // Update histogram bucket
      const bucketIdx = Math.min(4, Math.floor(score * 5));
      histogram[bucketIdx]++;

      return {
        candidate,
        score,
        nameScore,
        addrScore,
        isMatch: score >= threshold,
      };
    });

    scoredPool.sort((a, b) => b.score - a.score);
    const selected = scoredPool.filter((item) => item.isMatch).map((item) => item.candidate);

    return {
      source,
      selected,
      candidatesWithScores: scoredPool,
      candidates: pool,
    };
  });

  const hasGroundTruth = candidates.some((c) => Boolean(c.matchId));
  let precision: number | null = null;
  let recall: number | null = null;
  let f05: number | null = null;

  if (hasGroundTruth) {
    const entityScores = output.map(({ source, selected }) => {
      const truth = candidates.filter((candidate) => candidate.matchId === source.id);
      const correct = selected.filter((candidate) => candidate.matchId === source.id).length;
      const p = selected.length ? correct / selected.length : 1;
      const r = truth.length ? correct / truth.length : 1;
      const f = (1.25 * p * r) / (0.25 * p + r || 1);
      return { p, r, f };
    });
    const avg = (key: 'p' | 'r' | 'f') => entityScores.reduce((sum, score) => sum + score[key], 0) / (entityScores.length || 1);
    precision = avg('p');
    recall = avg('r');
    f05 = avg('f');
  }

  const totalCandidates = output.reduce((sum, row) => sum + row.candidates.length, 0);
  const totalMatches = output.reduce((sum, row) => sum + row.selected.length, 0);

  return {
    output,
    histogram,
    candidates: totalCandidates,
    matchesCount: totalMatches,
    hasGroundTruth,
    precision,
    recall,
    f05,
    source1Count: source1.length,
    otherSourcesCount: candidates.length,
  };
}

function App() {
  const [view, setView] = useState<View>('overview');
  const [mobileOpen, setMobileOpen] = useState(false);

  // Tunable Parameters
  const [threshold, setThreshold] = useState(0.52);
  const [nameWeight, setNameWeight] = useState(0.60);
  const [blockingStrategy, setBlockingStrategy] = useState<BlockingStrategy>('country');
  const [similarityMetric, setSimilarityMetric] = useState<SimilarityMetric>('hybrid');

  // Inspector Modal State
  const [inspectedEntity, setInspectedEntity] = useState<{ source: RecordRow; matches: CandidateMatch[] } | null>(null);

  // Persistent Runs
  const [runs, setRuns] = useState<Run[]>(() => {
    try {
      const saved = localStorage.getItem('resolve_lab_runs');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [run, setRun] = useState<Run | null>(runs[0] || null);
  const [submissionMade, setSubmissionMade] = useState(false);

  useEffect(() => {
    try {
      localStorage.setItem('resolve_lab_runs', JSON.stringify(runs));
    } catch {
      // Ignore
    }
  }, [runs]);

  // Dataset states
  const [customRecords, setCustomRecords] = useState<RecordRow[] | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFileMeta[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [groundTruthStatus, setGroundTruthStatus] = useState<string | null>(null);

  const activeRecords = useMemo(() => customRecords || syntheticRecords, [customRecords]);
  const isCustomDataset = Boolean(customRecords && customRecords.length > 0);

  const result = useMemo(
    () => runMatching(activeRecords, threshold, nameWeight, blockingStrategy, similarityMetric),
    [activeRecords, threshold, nameWeight, blockingStrategy, similarityMetric]
  );

  const navigate = (next: View) => {
    setView(next);
    setMobileOpen(false);
  };

  const handleFilesUpload = async (files: FileList | File[]) => {
    setUploadError(null);
    const fileArray = Array.from(files);
    if (!fileArray.length) return;

    try {
      const newRecords: RecordRow[] = [];
      const newMetas: UploadedFileMeta[] = [];

      for (const file of fileArray) {
        const text = await file.text();
        const { rows, source } = parseDelimitedText(text, file.name);
        if (rows.length === 0) {
          throw new Error(`File "${file.name}" has no valid records or is empty.`);
        }
        newRecords.push(...rows);
        newMetas.push({ name: file.name, rows: rows.length, source });
      }

      setCustomRecords((prev) => (prev ? [...prev, ...newRecords] : newRecords));
      setUploadedFiles((prev) => [...prev, ...newMetas]);
      setRun(null);
    } catch (err: unknown) {
      setUploadError(err instanceof Error ? err.message : 'Failed to parse uploaded files.');
    }
  };

  const handleGroundTruthUpload = async (files: FileList | File[]) => {
    setUploadError(null);
    const file = files[0];
    if (!file) return;

    try {
      const text = await file.text();
      const mapping = parseGroundTruthText(text);
      if (mapping.size === 0) throw new Error('No valid source1 -> match pairs found in ground truth file.');

      setCustomRecords((prev) => {
        const base = prev ? [...prev] : [...syntheticRecords];
        let attached = 0;
        const updated = base.map((r) => {
          if (r.source !== 'S1') {
            for (const [s1Id, matchTarget] of mapping.entries()) {
              if (matchTarget.split(',').includes(r.id) || matchTarget === r.id) {
                attached++;
                return { ...r, matchId: s1Id };
              }
            }
          }
          return r;
        });
        setGroundTruthStatus(`Loaded ground truth labels for ${attached} candidate entities from ${file.name}`);
        return updated;
      });
    } catch (err: unknown) {
      setUploadError(err instanceof Error ? err.message : 'Failed to parse ground truth file.');
    }
  };

  const handleResetToSynthetic = () => {
    setCustomRecords(null);
    setUploadedFiles([]);
    setUploadError(null);
    setGroundTruthStatus(null);
    setRun(null);
  };

  const executeRun = () => {
    const next: Run = {
      id: crypto.randomUUID(),
      name: `Experiment ${runs.length + 1} (${isCustomDataset ? 'Custom' : 'Synthetic'})`,
      score: result.f05 !== null ? Number(result.f05.toFixed(3)) : null,
      precision: result.precision !== null ? Number(result.precision.toFixed(3)) : null,
      recall: result.recall !== null ? Number(result.recall.toFixed(3)) : null,
      candidates: result.candidates,
      matchesCount: result.matchesCount,
      threshold,
      nameWeight,
      blockingStrategy,
      created: 'Just now',
      status: 'Complete',
    };
    setRun(next);
    setRuns((current) => [next, ...current]);
  };

  const download = (kind: 'matches' | 'candidates' | 'bundle') => {
    const matchesHeader = 'source1_entity_id\tmatched_entity_ids';
    const matchesRows = result.output.map(
      ({ source, selected }) => `${source.id}\t${selected.map((row) => row.id).join(',')}`
    );
    const matchesTsv = [matchesHeader, ...matchesRows].join('\n');

    const candHeader = 'source1_entity_id\tcandidate_entity_ids';
    const candRows = result.output.map(
      ({ source, candidates }) => `${source.id}\t${candidates.map((row) => row.id).join(',')}`
    );
    const candTsv = [candHeader, ...candRows].join('\n');

    if (kind === 'bundle') {
      const metadata = JSON.stringify(
        {
          datasetType: isCustomDataset ? 'custom_uploaded' : 'synthetic_demo',
          totalRecords: activeRecords.length,
          source1Count: result.source1Count,
          threshold,
          nameWeight,
          addressWeight: 1 - nameWeight,
          blockingStrategy,
          similarityMetric,
          metrics: {
            f05: result.f05,
            precision: result.precision,
            recall: result.recall,
            candidatePairs: result.candidates,
            totalMatches: result.matchesCount,
          },
          generatedAt: new Date().toISOString(),
        },
        null,
        2
      );

      const zipBlob = createZip([
        { name: 'matching_results.tsv', content: matchesTsv },
        { name: 'candidate_pairs.tsv', content: candTsv },
        { name: 'experiment_report.json', content: metadata },
      ]);

      const url = URL.createObjectURL(zipBlob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = 'submission_bundle.zip';
      anchor.click();
      URL.revokeObjectURL(url);
      setSubmissionMade(true);
      return;
    }

    const content = kind === 'matches' ? matchesTsv : candTsv;
    const blob = new Blob([content], { type: 'text/tab-separated-values' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${kind === 'matches' ? 'matching_results' : 'candidate_pairs'}.tsv`;
    anchor.click();
    URL.revokeObjectURL(url);
    if (kind === 'matches') setSubmissionMade(true);
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
        <div className="brand-row">
          <div className="brand-mark"><Network size={18} /></div>
          <span>Resolve Lab</span>
          <button className="mobile-close" onClick={() => setMobileOpen(false)}><X size={18} /></button>
        </div>
        <div className="workspace-switcher">
          <div className="workspace-avatar">ER</div>
          <div className="workspace-copy">
            <strong>Entity Resolution</strong>
            <span>{isCustomDataset ? 'Custom Dataset' : 'Demo Workspace'}</span>
          </div>
          <ChevronDown size={15} />
        </div>
        <nav className="primary-nav">
          <span className="nav-label">Workspace</span>
          {navItems.map(({ label, view: target, icon: Icon }) => (
            <button
              key={target}
              className={`nav-item ${view === target ? 'active' : ''}`}
              onClick={() => navigate(target)}
            >
              <Icon size={17} />
              {label}
              {view === target && <span className="active-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="integrity-card">
            <ShieldCheck size={16} color="#b7e77e" />
            <div>
              <strong>{isCustomDataset ? 'Uploaded Data Active' : 'Demo Data Active'}</strong>
              <span>{activeRecords.length} records · {result.source1Count} anchors</span>
            </div>
          </div>
          <button className="nav-item"><BookOpen size={17} />Documentation</button>
          <button className="nav-item"><Settings size={17} />Workspace settings</button>
          <div className="profile-row">
            <div className="profile-avatar">AS</div>
            <div>
              <strong>Alex Smith</strong>
              <span>Participant</span>
            </div>
            <span className="more-dots">•••</span>
          </div>
        </div>
      </aside>

      {mobileOpen && <button className="sidebar-scrim" onClick={() => setMobileOpen(false)} aria-label="Close navigation" />}

      <main className="main-content">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileOpen(true)}><Menu size={20} /></button>
          <div className="breadcrumbs">
            <span>Resolve Lab</span>
            <span>/</span>
            <strong>{navItems.find((item) => item.view === view)?.label}</strong>
          </div>
          <div className="topbar-actions">
            {isCustomDataset && (
              <span className="mode-badge-wrap">
                <Database size={13} /> Custom Data ({activeRecords.length} rows)
              </span>
            )}
            <button className="icon-button"><CircleHelp size={18} /></button>
            <button className="user-chip"><span className="online-dot" /> Alex <ChevronDown size={14} /></button>
          </div>
        </header>

        <div className="content-wrap">
          <ViewContent
            view={view}
            records={activeRecords}
            isCustomDataset={isCustomDataset}
            uploadedFiles={uploadedFiles}
            uploadError={uploadError}
            groundTruthStatus={groundTruthStatus}
            onFilesUpload={handleFilesUpload}
            onGroundTruthUpload={handleGroundTruthUpload}
            onResetToSynthetic={handleResetToSynthetic}
            result={result}
            threshold={threshold}
            setThreshold={setThreshold}
            nameWeight={nameWeight}
            setNameWeight={setNameWeight}
            blockingStrategy={blockingStrategy}
            setBlockingStrategy={setBlockingStrategy}
            similarityMetric={similarityMetric}
            setSimilarityMetric={setSimilarityMetric}
            onInspectEntity={(source, matches) => setInspectedEntity({ source, matches })}
            run={run}
            runs={runs}
            executeRun={executeRun}
            download={download}
            submissionMade={submissionMade}
            setSubmissionMade={setSubmissionMade}
            navigate={navigate}
          />
        </div>
      </main>

      {/* MATCH INSPECTOR MODAL */}
      {inspectedEntity && (
        <div className="modal-backdrop" onClick={() => setInspectedEntity(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>
                <Sliders size={18} />
                Entity Match Inspector
              </h3>
              <button className="modal-close-btn" onClick={() => setInspectedEntity(null)}>
                <X size={18} />
              </button>
            </div>
            <div className="modal-body">
              <div className="inspector-split">
                <div className="inspector-card-s1">
                  <span className="inspector-card-title">
                    <Database size={13} /> Source 1 Reference
                  </span>
                  <div className="inspector-s1-name">{inspectedEntity.source.name}</div>
                  <div className="inspector-s1-address">{inspectedEntity.source.address}</div>
                  <div className="inspector-badge-row">
                    <span className="tag-badge">ID: {inspectedEntity.source.id}</span>
                    <span className="tag-badge">Country: {inspectedEntity.source.country}</span>
                    <span className="tag-badge">{inspectedEntity.matches.filter((m) => m.isMatch).length} Matches</span>
                  </div>
                </div>

                <div>
                  <span className="inspector-card-title" style={{ marginBottom: 10 }}>
                    <Network size={13} /> Evaluated Candidates ({inspectedEntity.matches.length})
                  </span>
                  <div className="candidate-list">
                    {inspectedEntity.matches.length > 0 ? (
                      inspectedEntity.matches.map((item, idx) => (
                        <div className={`candidate-item ${item.isMatch ? 'is-match' : ''}`} key={idx}>
                          <div className="candidate-top">
                            <span className="candidate-name">{item.candidate.name}</span>
                            <span className={`source-badge ${item.candidate.source.toLowerCase()}`}>
                              {item.candidate.source}
                            </span>
                          </div>
                          <div className="candidate-address">{item.candidate.address}</div>
                          <div className="candidate-score-breakdown">
                            <span className="score-chip">
                              ID: {item.candidate.id}
                            </span>
                            <span className={`score-chip ${item.nameScore >= 0.7 ? 'high' : 'mid'}`}>
                              Name: {(item.nameScore * 100).toFixed(0)}%
                            </span>
                            <span className={`score-chip ${item.addrScore >= 0.7 ? 'high' : 'mid'}`}>
                              Addr: {(item.addrScore * 100).toFixed(0)}%
                            </span>
                            <span className="score-chip combined">
                              Score: {item.score.toFixed(3)}
                            </span>
                          </div>
                        </div>
                      ))
                    ) : (
                      <EmptyState text="No candidates passed blocking filters for this entity." />
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

type ContentProps = {
  view: View;
  records: RecordRow[];
  isCustomDataset: boolean;
  uploadedFiles: UploadedFileMeta[];
  uploadError: string | null;
  groundTruthStatus: string | null;
  onFilesUpload: (files: FileList | File[]) => void;
  onGroundTruthUpload: (files: FileList | File[]) => void;
  onResetToSynthetic: () => void;
  result: ReturnType<typeof runMatching>;
  threshold: number;
  setThreshold: (value: number) => void;
  nameWeight: number;
  setNameWeight: (value: number) => void;
  blockingStrategy: BlockingStrategy;
  setBlockingStrategy: (value: BlockingStrategy) => void;
  similarityMetric: SimilarityMetric;
  setSimilarityMetric: (value: SimilarityMetric) => void;
  onInspectEntity: (source: RecordRow, matches: CandidateMatch[]) => void;
  run: Run | null;
  runs: Run[];
  executeRun: () => void;
  download: (kind: 'matches' | 'candidates' | 'bundle') => void;
  submissionMade: boolean;
  setSubmissionMade: (value: boolean) => void;
  navigate: (view: View) => void;
};

function ViewContent(props: ContentProps) {
  const { view } = props;
  if (view === 'datasets') return <Datasets {...props} />;
  if (view === 'experiments') return <Experiments {...props} />;
  if (view === 'submissions') return <Submissions {...props} />;
  if (view === 'leaderboard') return <Leaderboard {...props} />;
  return <Overview {...props} />;
}

function PageHeader({ eyebrow, title, copy }: { eyebrow: string; title: ReactNode; copy: string }) {
  return (
    <section className="welcome-row">
      <div>
        <div className="eyebrow"><span className="live-pulse" /> {eyebrow}</div>
        <h1>{title}</h1>
        <p className="lede">{copy}</p>
      </div>
      <div className="deadline-card">
        <div className="deadline-icon"><Zap size={17} /></div>
        <div>
          <span>Pipeline Status</span>
          <strong>Active & Ready</strong>
          <small>Interactive matching & exports</small>
        </div>
      </div>
    </section>
  );
}

function Overview({
  records,
  isCustomDataset,
  uploadedFiles,
  uploadError,
  groundTruthStatus,
  onFilesUpload,
  onGroundTruthUpload,
  onResetToSynthetic,
  result,
  run,
  runs,
  navigate,
}: ContentProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const groundTruthInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragover' || e.type === 'dragenter') setIsDragging(true);
    else if (e.type === 'dragleave') setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFilesUpload(e.dataTransfer.files);
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesUpload(e.target.files);
    }
  };

  return (
    <>
      <PageHeader
        eyebrow={isCustomDataset ? 'Custom Dataset Active' : 'Competition Practice Lab'}
        title={<>Business Entity<br /><em>Resolution Studio</em></>}
        copy="Upload multi-source business datasets, tune candidate blocking & similarity thresholds, inspect match diffs, and export complete competition bundles."
      />

      <div className="metric-grid">
        <Metric icon={<Database size={17} />} label="Total Records" value={String(records.length)} foot={isCustomDataset ? 'Uploaded datasets' : 'Across 3 sources'} />
        <Metric icon={<Network size={17} />} label="Source 1 Anchors" value={String(result.source1Count)} foot="Reference companies" />
        <Metric
          icon={<BarChart3 size={17} />}
          label={result.hasGroundTruth ? 'Current F₀.₅' : 'Candidate Pairs'}
          value={result.hasGroundTruth && run && run.score !== null ? run.score.toFixed(3) : String(result.candidates)}
          foot={result.hasGroundTruth ? (run ? 'Latest experiment' : 'Run first test') : 'Pairs generated'}
        />
        <Metric
          icon={<FileCheck2 size={17} />}
          label="Submission Status"
          value={run ? 'Ready' : 'Not started'}
          foot={run ? `${result.matchesCount} matches resolved` : 'Awaiting experiment'}
        />
      </div>

      <div className="workspace-grid">
        <div className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <span className="section-kicker">01 / Dataset</span>
              <h2>{isCustomDataset ? 'Custom Data Loaded' : 'Upload or explore data'}</h2>
              <p>Supports .tsv, .csv files (e.g. source1.tsv, source2.tsv, source3.tsv).</p>
            </div>
            <div className="heading-icon">
              {isCustomDataset ? <Database size={20} /> : <Sparkles size={20} />}
            </div>
          </div>

          <div
            className={`dropzone ${isDragging ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{ minHeight: 220 }}
          >
            <input
              type="file"
              multiple
              accept=".tsv,.csv,.txt"
              ref={fileInputRef}
              className="file-input-hidden"
              onChange={handleInputChange}
            />

            <input
              type="file"
              accept=".tsv,.csv,.txt"
              ref={groundTruthInputRef}
              className="file-input-hidden"
              onChange={(e) => e.target.files && onGroundTruthUpload(e.target.files)}
            />

            <div className="upload-icon">
              <Upload size={22} />
            </div>

            <strong>
              {isCustomDataset ? `${records.length} records ready` : 'Drag and drop your dataset files here'}
            </strong>
            <span>Drop train_source1.tsv, source2.csv, or unified datasets</span>

            <div className="dropzone-actions" onClick={(e) => e.stopPropagation()}>
              <button className="primary-button" onClick={() => fileInputRef.current?.click()}>
                <Upload size={14} /> Upload Dataset Files
              </button>
              <button className="outline-button" onClick={() => groundTruthInputRef.current?.click()} title="Upload ground truth mapping file">
                <FileCheck2 size={14} /> Upload Ground Truth
              </button>
              <button className="browse-button" onClick={() => navigate('datasets')}>
                <Search size={14} /> Browse Dataset
              </button>
              {isCustomDataset && (
                <button className="outline-button" onClick={onResetToSynthetic} title="Reset to default practice dataset">
                  <RotateCcw size={14} /> Reset to Demo
                </button>
              )}
            </div>

            <small>Accepts tab-separated (.tsv) or comma-separated (.csv) files</small>
          </div>

          {uploadError && (
            <div className="error-banner">
              <AlertCircle size={15} /> {uploadError}
            </div>
          )}

          {groundTruthStatus && (
            <div className="success-banner" style={{ marginTop: 10 }}>
              <Check size={15} /> {groundTruthStatus}
            </div>
          )}

          {uploadedFiles.length > 0 && (
            <div className="file-chips-container">
              {uploadedFiles.map((file, idx) => (
                <div className="file-chip" key={idx}>
                  <FileText size={12} />
                  <span>{file.name}</span>
                  <small>({file.rows} rows · {file.source})</small>
                </div>
              ))}
            </div>
          )}

          <div className="supported-files">
            <span><Check size={13} /> Multi-file TSV/CSV</span>
            <span><Check size={13} /> Auto source detection</span>
            <span><Check size={13} /> Match Inspector Diff</span>
            <span><Check size={13} /> ZIP Submission Bundle</span>
          </div>
        </div>

        <div className="right-stack">
          <div className="panel progress-panel">
            <div className="panel-heading compact">
              <div>
                <span className="section-kicker">02 / Pipeline</span>
                <h2>Readiness checklist</h2>
              </div>
            </div>
            <div className="progress-line">
              <div className="progress-label">
                <span>Overall pipeline readiness</span>
                <strong>{run ? '100%' : isCustomDataset ? '60%' : '25%'}</strong>
              </div>
              <div className="progress-track">
                <div className="progress-value" style={{ width: run ? '100%' : isCustomDataset ? '60%' : '25%' }} />
              </div>
            </div>
            <div className="checklist">
              <ProgressItem done label="Workspace initialized" />
              <ProgressItem done={Boolean(records.length)} label={`Dataset ready (${records.length} records)`} />
              <ProgressItem done={Boolean(run)} label="Matching experiment executed" />
              <ProgressItem done={Boolean(run)} label="Submission format validated" />
            </div>
            <button className="primary-button" onClick={() => navigate('experiments')}>
              <Play size={16} fill="currentColor" /> {run ? 'Review Experiment' : 'Start Experiment'} <ArrowUpRight size={15} />
            </button>
          </div>

          <div className="tip-card">
            <div className="tip-icon"><Sparkles size={18} /></div>
            <div>
              <strong>F₀.₅ Precision Bias</strong>
              <p>F₀.₅ score weights precision 2× heavier than recall. Guard against false positive merges with tight blocking.</p>
            </div>
          </div>
        </div>
      </div>

      <div className="bottom-grid">
        <div className="panel files-panel">
          <div className="panel-heading compact">
            <div>
              <span className="section-kicker">Latest activity</span>
              <h2>Experiment history</h2>
            </div>
          </div>
          {runs.length ? (
            runs.slice(0, 4).map((item) => (
              <div className="table-row" key={item.id}>
                <div className="file-name">
                  <div className="file-icon"><FlaskConical size={15} /></div>
                  <span>
                    <strong>{item.name}</strong>
                    <small>{item.created} · {item.candidates} candidates · {item.matchesCount} matches (cutoff {item.threshold.toFixed(2)})</small>
                  </span>
                </div>
                <strong className="file-type">{item.score !== null ? item.score.toFixed(3) : `${item.matchesCount} matches`}</strong>
                <span className="status-pill"><span />{item.status}</span>
              </div>
            ))
          ) : (
            <EmptyState text="No experiments run yet. Adjust thresholds and run your first matching experiment." />
          )}
        </div>

        <div className="panel quick-panel">
          <span className="section-kicker">Challenge Guidelines</span>
          <h2>Key Rules</h2>
          <div className="rule-list">
            <Rule icon={<Search size={16} />} title="Block before matching" text="Candidate blocking filters out obvious non-matches early." />
            <Rule icon={<FileCheck2 size={16} />} title="One row per S1 entity" text="Output format strictly requires one row per Source 1 ID." />
            <Rule icon={<BarChart3 size={16} />} title="F₀.₅ scoring" text="Evaluates precision with twice the importance of recall." />
          </div>
          <button className="learn-button" onClick={() => navigate('submissions')}>
            Open submission builder <ArrowUpRight size={15} />
          </button>
        </div>
      </div>
    </>
  );
}

function Datasets({
  records,
  isCustomDataset,
  onFilesUpload,
  onResetToSynthetic,
  onInspectEntity,
  result,
  navigate,
}: ContentProps) {
  const [filterSource, setFilterSource] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'name' | 'id' | 'country'>('name');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(24);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const sources = useMemo(() => {
    const set = new Set(records.map((r) => r.source));
    return ['ALL', ...Array.from(set)];
  }, [records]);

  const filtered = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    const rows = records.filter((r) => {
      const matchesSource = filterSource === 'ALL' || r.source === filterSource;
      const matchesQuery =
        !q ||
        r.name.toLowerCase().includes(q) ||
        r.address.toLowerCase().includes(q) ||
        r.id.toLowerCase().includes(q) ||
        r.country.toLowerCase().includes(q);
      return matchesSource && matchesQuery;
    });

    rows.sort((a, b) => {
      const valA = a[sortBy].toLowerCase();
      const valB = b[sortBy].toLowerCase();
      return sortOrder === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
    });

    return rows;
  }, [records, filterSource, searchQuery, sortBy, sortOrder]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const paginatedRows = useMemo(() => {
    const start = (page - 1) * pageSize;
    return filtered.slice(start, start + pageSize);
  }, [filtered, page, pageSize]);

  return (
    <>
      <PageHeader
        eyebrow="Data explorer"
        title={<>Explore & inspect<br /><em>your dataset.</em></>}
        copy="Search through raw records, inspect noise across sources, or click any Source 1 record to inspect candidate diffs."
      />

      <div className="panel files-panel">
        <div className="panel-heading compact">
          <div>
            <span className="section-kicker">
              {isCustomDataset ? 'Custom Dataset' : 'Demo Dataset'} · {records.length} total rows
            </span>
            <h2>Source records</h2>
          </div>
          <div className="download-actions">
            <input
              type="file"
              multiple
              accept=".tsv,.csv,.txt"
              ref={fileInputRef}
              className="file-input-hidden"
              onChange={(e) => e.target.files && onFilesUpload(e.target.files)}
            />
            <button className="primary-button" onClick={() => fileInputRef.current?.click()}>
              <Upload size={15} /> Upload Files
            </button>
            {isCustomDataset && (
              <button className="outline-button" onClick={onResetToSynthetic}>
                <RotateCcw size={15} /> Reset Demo
              </button>
            )}
            <button className="outline-button" onClick={() => navigate('experiments')}>
              <FlaskConical size={15} /> Open Experiment Studio
            </button>
          </div>
        </div>

        <div className="dataset-toolbar">
          <div className="dataset-search-wrap">
            <Search size={15} />
            <input
              type="text"
              placeholder="Search by company name, ID, address, country..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setPage(1);
              }}
              className="dataset-search-input"
            />
          </div>

          <div className="dataset-filter-tabs">
            {sources.map((src) => (
              <button
                key={src}
                className={`filter-tab ${filterSource === src ? 'active' : ''}`}
                onClick={() => {
                  setFilterSource(src);
                  setPage(1);
                }}
              >
                {src === 'ALL' ? `All (${records.length})` : `${src} (${records.filter((r) => r.source === src).length})`}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <select
              className="select-control"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'name' | 'id' | 'country')}
            >
              <option value="name">Sort by Name</option>
              <option value="id">Sort by ID</option>
              <option value="country">Sort by Country</option>
            </select>
            <button
              className="page-btn"
              onClick={() => setSortOrder((o) => (o === 'asc' ? 'desc' : 'asc'))}
              title="Toggle Ascending/Descending"
            >
              {sortOrder === 'asc' ? '↑ A-Z' : '↓ Z-A'}
            </button>
          </div>
        </div>

        <div className="data-grid" style={{ maxHeight: '600px', overflowY: 'auto' }}>
          {paginatedRows.length > 0 ? (
            paginatedRows.map((record) => {
              const s1Output = record.source === 'S1' ? result.output.find((o) => o.source.id === record.id) : null;
              return (
                <div
                  className={`record-card ${record.source === 'S1' ? 'clickable-row' : ''}`}
                  key={record.id}
                  onClick={() => {
                    if (s1Output) {
                      onInspectEntity(record, s1Output.candidatesWithScores);
                    }
                  }}
                  title={record.source === 'S1' ? 'Click to inspect candidate diffs' : undefined}
                >
                  <div className={`source-badge ${record.source.toLowerCase()}`}>{record.source}</div>
                  <div className="record-copy">
                    <strong>{record.name}</strong>
                    <span>{record.address || 'No address specified'}</span>
                    <small>{record.country} · {record.id}</small>
                  </div>
                  {record.source === 'S1' && (
                    <button
                      className="outline-button"
                      style={{ padding: '4px 8px', fontSize: 11, marginLeft: 'auto' }}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (s1Output) onInspectEntity(record, s1Output.candidatesWithScores);
                      }}
                    >
                      <Eye size={12} /> Inspect
                    </button>
                  )}
                  {record.matchId && (
                    <span title={`Ground truth match: ${record.matchId}`} style={{ display: 'inline-flex', alignItems: 'center', gap: 3, fontSize: 11, color: '#387346' }}>
                      <Check size={14} className="record-check" /> {record.matchId}
                    </span>
                  )}
                </div>
              );
            })
          ) : (
            <EmptyState text="No records match your search query." />
          )}
        </div>

        <div className="pagination-footer">
          <span style={{ fontSize: 12, color: '#556c64' }}>
            Showing {filtered.length === 0 ? 0 : (page - 1) * pageSize + 1}–{Math.min(page * pageSize, filtered.length)} of {filtered.length} records
          </span>
          <div className="pagination-controls">
            <select
              className="select-control"
              value={pageSize}
              onChange={(e) => {
                setPageSize(Number(e.target.value));
                setPage(1);
              }}
            >
              <option value={12}>12 / page</option>
              <option value={24}>24 / page</option>
              <option value={48}>48 / page</option>
              <option value={96}>96 / page</option>
            </select>
            <button className="page-btn" disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>
              <ChevronLeft size={14} /> Prev
            </button>
            <span style={{ fontSize: 12, fontWeight: 600, padding: '0 6px' }}>
              {page} / {totalPages}
            </span>
            <button className="page-btn" disabled={page >= totalPages} onClick={() => setPage((p) => Math.min(totalPages, p + 1))}>
              Next <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

function Experiments({
  records,
  isCustomDataset,
  result,
  threshold,
  setThreshold,
  nameWeight,
  setNameWeight,
  blockingStrategy,
  setBlockingStrategy,
  similarityMetric,
  setSimilarityMetric,
  onInspectEntity,
  run,
  executeRun,
  download,
}: ContentProps) {
  const maxHistogramCount = Math.max(1, ...result.histogram);
  const cutoffBucketIdx = Math.min(4, Math.floor(threshold * 5));

  return (
    <>
      <PageHeader
        eyebrow="Experiment studio"
        title={<>Tune your<br /><em>matcher.</em></>}
        copy={`Adjust similarity thresholds, candidate blocking keys, and weights across ${records.length} records.`}
      />

      <div className="experiment-grid">
        <div className="panel config-panel">
          <span className="section-kicker">Configuration</span>
          <h2>Similarity matcher</h2>
          <p>Fine-tune feature weights, algorithms, and candidate blocking strategies.</p>

          <label className="slider-label">
            <span>Match Threshold (Cutoff)</span>
            <strong>{threshold.toFixed(2)}</strong>
          </label>
          <input
            type="range"
            min="0.20"
            max="0.95"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
          <div className="slider-scale">
            <span>More Recall (lower)</span>
            <span>More Precision (higher)</span>
          </div>

          <label className="slider-label" style={{ marginTop: 14 }}>
            <span>Name Weight: {(nameWeight * 100).toFixed(0)}% · Address: {((1 - nameWeight) * 100).toFixed(0)}%</span>
            <strong>{nameWeight.toFixed(2)}</strong>
          </label>
          <input
            type="range"
            min="0.10"
            max="0.90"
            step="0.05"
            value={nameWeight}
            onChange={(e) => setNameWeight(Number(e.target.value))}
          />

          <div style={{ marginTop: 14 }}>
            <label className="slider-label">
              <span>Blocking Strategy</span>
            </label>
            <select
              className="select-control"
              style={{ width: '100%', padding: '8px 12px' }}
              value={blockingStrategy}
              onChange={(e) => setBlockingStrategy(e.target.value as BlockingStrategy)}
            >
              <option value="country">Country Blocking (Default)</option>
              <option value="country_prefix">Country + First Letter of Name</option>
              <option value="prefix_3">Prefix (First 3 Letters)</option>
              <option value="exhaustive">Exhaustive (All-to-All Comparison)</option>
            </select>
          </div>

          <div style={{ marginTop: 12 }}>
            <label className="slider-label">
              <span>Similarity Algorithm</span>
            </label>
            <select
              className="select-control"
              style={{ width: '100%', padding: '8px 12px' }}
              value={similarityMetric}
              onChange={(e) => setSimilarityMetric(e.target.value as SimilarityMetric)}
            >
              <option value="hybrid">Hybrid (Character Overlap + Token Jaccard)</option>
              <option value="token_jaccard">Token Jaccard (Word Overlap)</option>
              <option value="levenshtein">Levenshtein (Edit Distance)</option>
            </select>
          </div>

          <div className="config-tags" style={{ marginTop: 14 }}>
            <span>{blockingStrategy}</span>
            <span>Name {(nameWeight * 100).toFixed(0)}%</span>
            <span>Addr {((1 - nameWeight) * 100).toFixed(0)}%</span>
            <span>{isCustomDataset ? 'Custom data' : 'Demo data'}</span>
          </div>

          <button className="primary-button" onClick={executeRun}>
            <Play size={16} fill="currentColor" /> Run experiment
          </button>
        </div>

        <div className="panel results-panel">
          <span className="section-kicker">Live preview</span>
          <h2>Validation metrics</h2>

          <div className="score-display">
            <strong>{result.f05 !== null ? result.f05.toFixed(3) : '—'}</strong>
            <span>{result.hasGroundTruth ? 'F₀.₅ validation score' : 'Unlabeled evaluation mode'}</span>
          </div>

          <div className="mini-metrics">
            <Metric
              icon={<Check size={17} />}
              label="Precision"
              value={result.precision !== null ? `${(result.precision * 100).toFixed(0)}%` : 'N/A'}
              foot={result.hasGroundTruth ? 'Selected correct matches' : 'Upload ground truth to evaluate'}
            />
            <Metric
              icon={<Network size={17} />}
              label="Recall"
              value={result.recall !== null ? `${(result.recall * 100).toFixed(0)}%` : 'N/A'}
              foot={result.hasGroundTruth ? 'True entities discovered' : 'Calculated on submission'}
            />
          </div>

          {/* HISTOGRAM SCORE DISTRIBUTION */}
          <div className="histogram-card">
            <div className="histogram-title-row">
              <span>Candidate Similarity Distribution</span>
              <small>Threshold: {threshold.toFixed(2)}</small>
            </div>
            <div className="histogram-bars">
              {result.histogram.map((count, idx) => {
                const heightPercent = maxHistogramCount ? (count / maxHistogramCount) * 100 : 0;
                const isAbove = idx >= cutoffBucketIdx;
                const labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'];
                return (
                  <div className="histogram-col" key={idx}>
                    <span className="histogram-count">{count}</span>
                    <div className="histogram-bar-wrap">
                      <div
                        className={`histogram-bar ${isAbove ? 'above-threshold' : ''}`}
                        style={{ height: `${Math.max(8, heightPercent)}%` }}
                        title={`${count} pairs in ${labels[idx]}`}
                      />
                    </div>
                    <span className="histogram-label">{labels[idx]}</span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="result-line">
            <span>Source 1 Reference Entities</span>
            <strong>{result.source1Count}</strong>
          </div>
          <div className="result-line">
            <span>Candidate Pairs Generated</span>
            <strong>{result.candidates}</strong>
          </div>
          <div className="result-line">
            <span>Matches Above Threshold</span>
            <strong>{result.matchesCount}</strong>
          </div>
        </div>
      </div>

      <div className="panel files-panel">
        <div className="panel-heading compact">
          <div>
            <span className="section-kicker">Output preview & Inspector</span>
            <h2>Click any entity to inspect side-by-side diffs</h2>
          </div>
          <div className="download-actions">
            <button className="primary-button" onClick={() => download('bundle')}>
              <Archive size={15} /> Export ZIP Bundle
            </button>
            <button className="outline-button" onClick={() => download('candidates')}>
              <ArrowDownToLine size={15} /> candidate_pairs.tsv
            </button>
            <button className="outline-button" onClick={() => download('matches')}>
              <ArrowDownToLine size={15} /> matching_results.tsv
            </button>
          </div>
        </div>

        <div className="output-preview">
          <code>source1_entity_id{`\t`}matched_entity_ids</code>
          {result.output.slice(0, 6).map(({ source, selected, candidatesWithScores }) => (
            <code
              key={source.id}
              className="clickable-row"
              style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
              onClick={() => onInspectEntity(source, candidatesWithScores)}
              title="Click to inspect side-by-side"
            >
              <span>{source.id}{`\t`}{selected.map((r) => r.id).join(',') || '(empty)'}</span>
              <span style={{ fontSize: 11, color: '#387346', fontWeight: 600 }}>Inspect ↗</span>
            </code>
          ))}
          {result.output.length > 6 && <code>... ({result.output.length - 6} more rows ready for export)</code>}
        </div>

        {run && (
          <div className="success-banner">
            <Check size={16} /> {run.name} generated {result.matchesCount} resolved matches across {result.source1Count} S1 entities.
          </div>
        )}
      </div>
    </>
  );
}

function Submissions({
  result,
  run,
  download,
  submissionMade,
}: ContentProps) {
  return (
    <>
      <PageHeader
        eyebrow="Submission center"
        title={<>Package your<br /><em>results.</em></>}
        copy="Validate the official tab-separated outputs and download the full competition bundle."
      />

      <div className="submission-grid">
        <div className="panel files-panel">
          <div className="panel-heading compact">
            <div>
              <span className="section-kicker">Leaderboard file</span>
              <h2>matching_results.tsv</h2>
              <p>One row for every Source 1 entity. Format: source1_entity_id\tmatched_entity_ids</p>
            </div>
            <FileCheck2 color="#62a16b" />
          </div>
          <div className="validation-card">
            <div className="validation-icon"><Check size={20} /></div>
            <div>
              <strong>{result.source1Count > 0 ? 'Format validation passed' : 'No Source 1 entities'}</strong>
              <span>
                {run ? `${run.name} active · ` : ''}
                {result.output.length} Source 1 entities with {result.matchesCount} resolved matches.
              </span>
            </div>
          </div>
          <button
            className="primary-button"
            onClick={() => download('matches')}
          >
            <ArrowDownToLine size={16} /> Download matching_results.tsv
          </button>
        </div>

        <div className="panel files-panel">
          <div className="panel-heading compact">
            <div>
              <span className="section-kicker">Companion file</span>
              <h2>candidate_pairs.tsv</h2>
              <p>Documents all candidate pairs generated by blocking before thresholding.</p>
            </div>
            <Network color="#62a16b" />
          </div>
          <div className="validation-card">
            <div className="validation-icon"><ShieldCheck size={20} /></div>
            <div>
              <strong>Blocking coverage captured</strong>
              <span>{result.candidates} candidate pair relationships documented.</span>
            </div>
          </div>
          <button className="outline-button wide-button" onClick={() => download('candidates')}>
            <ArrowDownToLine size={16} /> Download candidate_pairs.tsv
          </button>
        </div>
      </div>

      <div className="panel files-panel" style={{ marginTop: 20 }}>
        <div className="panel-heading compact">
          <div>
            <span className="section-kicker">Complete Archive</span>
            <h2>Download All as Submission Bundle (.zip)</h2>
            <p>Contains matching_results.tsv, candidate_pairs.tsv, and experiment_report.json.</p>
          </div>
          <button className="primary-button" onClick={() => download('bundle')}>
            <Archive size={16} /> Download submission_bundle.zip
          </button>
        </div>
      </div>

      {submissionMade && (
        <div className="success-banner large" style={{ marginTop: 20 }}>
          <Check size={17} /> Submission files generated and downloaded! Both files strictly follow the tab-separated competition structure.
        </div>
      )}
    </>
  );
}

function Leaderboard({ runs }: ContentProps) {
  const rows = runs.length
    ? runs
    : [
        {
          id: 'sample',
          name: 'Baseline reference run',
          score: 0.714,
          precision: 0.8,
          recall: 0.667,
          candidates: 31,
          matchesCount: 16,
          threshold: 0.52,
          nameWeight: 0.6,
          blockingStrategy: 'country' as BlockingStrategy,
          created: 'Challenge default',
          status: 'Complete' as const,
        },
      ];

  return (
    <>
      <PageHeader
        eyebrow="Experiment leaderboard"
        title={<>Measure what<br /><em>matters.</em></>}
        copy="Compare runs across candidate reduction, matches count, and F₀.₅ scores."
      />
      <div className="panel files-panel leaderboard-panel">
        <div className="table-head leaderboard-head">
          <span>Rank / experiment</span>
          <span>Precision</span>
          <span>Recall</span>
          <span>F₀.₅ / Matches</span>
        </div>
        {rows.map((item, index) => (
          <div className="leader-row" key={item.id}>
            <div className="rank-name">
              <strong>#{index + 1}</strong>
              <div className="file-icon"><BarChart3 size={15} /></div>
              <span>
                <b>{item.name}</b>
                <small>{item.created} · {item.candidates} candidates · {item.matchesCount} matches · cutoff {item.threshold.toFixed(2)}</small>
              </span>
            </div>
            <span>{item.precision !== null ? `${(item.precision * 100).toFixed(1)}%` : '—'}</span>
            <span>{item.recall !== null ? `${(item.recall * 100).toFixed(1)}%` : '—'}</span>
            <strong className="leader-score">
              {item.score !== null ? item.score.toFixed(3) : `${item.matchesCount} m`}
            </strong>
          </div>
        ))}
      </div>
    </>
  );
}

function Metric({ icon, label, value, foot }: { icon: ReactNode; label: string; value: string; foot: string }) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <div className="metric-icon">{icon}</div>
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
      <small>{foot}</small>
    </div>
  );
}

function ProgressItem({ label, done = false }: { label: string; done?: boolean }) {
  return (
    <div className={`check-item ${done ? 'done' : ''}`}>
      <div className="check-circle">{done && <Check size={12} />}</div>
      <span>{label}</span>
      {done && <small>Complete</small>}
    </div>
  );
}

function Rule({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return (
    <div className="rule-item">
      <div className="rule-icon">{icon}</div>
      <div>
        <strong>{title}</strong>
        <span>{text}</span>
      </div>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="empty-state">{text}</div>;
}

export default App;