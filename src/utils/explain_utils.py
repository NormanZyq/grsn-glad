# component map
component_map_str = """
	0	O
	1	C
	2	N
	3	F
	4	Cl
	5	S
	6	Br
	7	Si
	8	Na
	9	I
	10	Hg
	11	B
	12	K
	13	P
	14	Au
	15	Cr
	16	Sn
	17	Ca
	18	Cd
	19	Zn
	20	V
	21	As
	22	Li
	23	Cu
	24	Co
	25	Ag
	26	Se
	27	Pt
	28	Al
	29	Bi
	30	Sb
	31	Ba
	32	Fe
	33	H
	34	Ti
	35	Tl
	36	Sr
	37	In
	38	Dy
	39	Ni
	40	Be
	41	Mg
	42	Nd
	43	Pd
	44	Mn
	45	Zr
	46	Pb
	47	Yb
	48	Mo
	49	Ge
	50	Ru
	51	Eu
	52	Sc
	53	Gd
""".strip()
component_map = {int(line.split()[0]): line.split()[1] for line in component_map_str.split('\n')}

edge_label_map = {0: '-', 1: '=', 2: ':', 3: '#'}