import argparse
import glob
import os
import xml.etree.ElementTree as ET
import numpy as np


def parse_tm(path, node_ids):
    root = ET.parse(path).getroot()
    tm = root.find('.//IntraTM')
    if tm is None:
        raise ValueError(f'No IntraTM in {path}')
    idx = {n: i for i, n in enumerate(node_ids)}
    out = np.zeros((len(node_ids), len(node_ids)), dtype=np.float32)
    seen_src = set()
    for src in tm.findall('src'):
        sid = src.attrib['id']
        if sid not in idx:
            raise ValueError(f'Unexpected source node {sid} in {path}')
        i = idx[sid]
        seen_src.add(sid)
        for dst in src.findall('dst'):
            did = dst.attrib['id']
            if did not in idx:
                raise ValueError(f'Unexpected destination node {did} in {path}')
            out[i, idx[did]] = float(dst.text or 0.0)
    if len(seen_src) != len(node_ids):
        raise ValueError(f'{path}: expected {len(node_ids)} source nodes, got {len(seen_src)}')
    return out


def convert_tm(files, out_path):
    if not files:
        raise FileNotFoundError('No traffic-matrix XML files found')
    files = sorted(files)
    root = ET.parse(files[0]).getroot()
    tm = root.find('.//IntraTM')
    node_ids = [s.attrib['id'] for s in tm.findall('src')]
    arr = np.empty((len(files), len(node_ids), len(node_ids)), dtype=np.float32)
    for k, path in enumerate(files):
        arr[k] = parse_tm(path, node_ids)
        if (k + 1) % 1000 == 0:
            print(f'  parsed {k + 1}/{len(files)}')
    np.save(out_path, arr)
    return node_ids, arr


def make_geant_edges(topology_xml, node_ids, out_path):
    root = ET.parse(topology_xml).getroot()
    idx = {n: i for i, n in enumerate(node_ids)}
    seen = set()
    edges = []
    for link in root.findall('.//links/link'):
        f = link.find('from')
        t = link.find('to')
        if f is None or t is None:
            continue
        a = idx[f.attrib['node']]
        b = idx[t.attrib['node']]
        e = tuple(sorted((a, b)))
        if e not in seen:
            seen.add(e)
            edges.append(e)
    with open(out_path, 'w') as f:
        for a, b in edges:
            f.write(f'{a} {b}\n')
    return edges


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data', help='folder containing the extracted datasets')
    ap.add_argument('--out', default='.', help='output folder')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    abilene_files = glob.glob(os.path.join(args.data_root, 'abilene-TM', 'TM', '**', '*.xml'), recursive=True)
    # Exclude anything under topo; the pattern above already points under TM.
    geant_files = glob.glob(os.path.join(args.data_root, 'traffic-matrices', '*.xml'))
    geant_topo = os.path.join(args.data_root, 'topology-anonymised.xml')

    print(f'Abilene XML files: {len(abilene_files)}')
    a_nodes, a = convert_tm(abilene_files, os.path.join(args.out, 'abilene.npy'))
    print('Abilene:', a.shape, a_nodes)

    print(f'GEANT XML files: {len(geant_files)}')
    g_nodes, g = convert_tm(geant_files, os.path.join(args.out, 'geant.npy'))
    print('GEANT:', g.shape, g_nodes)

    edges = make_geant_edges(geant_topo, g_nodes, os.path.join(args.out, 'geant_edges.txt'))
    print(f'GEANT topology: {len(g_nodes)} nodes, {len(edges)} unique undirected edges')

    with open(os.path.join(args.out, 'dataset_info.txt'), 'w') as f:
        f.write(f'Abilene shape: {a.shape}\n')
        f.write('Abilene nodes: ' + ', '.join(a_nodes) + '\n')
        f.write('Abilene sampling: 5 minutes\n')
        f.write('Units: kbps\n\n')
        f.write(f'GEANT shape: {g.shape}\n')
        f.write('GEANT nodes: ' + ', '.join(g_nodes) + '\n')
        f.write('GEANT sampling: 15 minutes\n')
        f.write('Units: kbps\n')
        f.write(f'GEANT unique undirected edges: {len(edges)}\n')
