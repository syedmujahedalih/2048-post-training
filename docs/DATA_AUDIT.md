# Demonstration inspection sheet

Twenty reproducibly sampled training examples. No test examples are inspected for prompt tuning.

Each target is a legal move and is recomputed using the pinned teacher.

## train-1023-72 (early, teacher)

```text
   0    2    4    2
   0    4    2    8
  32    4    2    4
  64   16    8    4
```

Target: **DOWN**. Legal: UP, DOWN, LEFT.

Teacher action values (heuristic units): `{"UP": 1551824.12, "DOWN": 1578688.12, "LEFT": 1576361.25}`

## train-1041-72 (early, priority)

```text
   4    2    0    0
   8    8    0    2
   4   16    0    0
  64   32   16    2
```

Target: **DOWN**. Legal: UP, DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"UP": 1555403.0, "DOWN": 1565353.88, "LEFT": 1558621.12, "RIGHT": 1557987.0}`

## train-1042-68 (early, teacher)

```text
   2    4   64   64
   0    0    4    8
   0    0    8    2
   0    0    2    0
```

Target: **RIGHT**. Legal: DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"DOWN": 1563578.88, "LEFT": 1565007.12, "RIGHT": 1583210.0}`

## train-1031-32 (early, priority)

```text
   0    0    4    0
   4    0    4    0
  16    8    4    0
   2   16    8    8
```

Target: **DOWN**. Legal: UP, DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"UP": 1566605.62, "DOWN": 1589554.88, "LEFT": 1573412.75, "RIGHT": 1574218.0}`

## train-1061-21 (early, priority)

```text
   2    0    0    0
   4    2    2    0
   8    4    2    0
  16    8    4    2
```

Target: **DOWN**. Legal: UP, DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"UP": 1599490.12, "DOWN": 1600193.12, "LEFT": 1600193.12, "RIGHT": 1599490.0}`

## train-1042-1299 (late, teacher)

```text
2048   64   16    4
 512   32    2    4
 128    8    4    0
  16    4    2    0
```

Target: **UP**. Legal: UP, DOWN, RIGHT.

Teacher action values (heuristic units): `{"UP": 1411037.88, "DOWN": 1406052.12, "RIGHT": 1294343.38}`

## train-1056-17 (early, priority)

```text
   2    0    0    0
   0    2    0    0
   2    8    4    4
   8    2    8    8
```

Target: **RIGHT**. Legal: UP, DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"UP": 1590207.12, "DOWN": 1594625.38, "LEFT": 1591813.25, "RIGHT": 1596491.38}`

## train-1052-52 (early, teacher)

```text
  64   16    8    2
   4    8    4    0
   4    2    0    0
   2    0    2    0
```

Target: **LEFT**. Legal: UP, DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"UP": 1587434.0, "DOWN": 1571042.38, "LEFT": 1587552.5, "RIGHT": 1587502.12}`

## train-1070-30 (early, random)

```text
   2    8    4    0
   4   32    2    0
  16    0    0    0
   2    0    2    0
```

Target: **DOWN**. Legal: UP, DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"UP": 1551529.88, "DOWN": 1553870.38, "LEFT": 1549482.25, "RIGHT": 1548137.62}`

## train-1029-1522 (late, teacher)

```text
2048 1024   64   32
 128   16    8    4
   8    8    4    2
   4    0    2    0
```

Target: **LEFT**. Legal: DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"DOWN": 914726.94, "LEFT": 1391138.12, "RIGHT": 1390557.38}`

## train-1061-228 (middle, priority)

```text
   2    4    2    2
   8   16    8    4
 128    2   16    8
 256   32    4   16
```

Target: **RIGHT**. Legal: LEFT, RIGHT.

Teacher action values (heuristic units): `{"LEFT": 1485011.75, "RIGHT": 1487728.25}`

## train-1019-280 (middle, teacher)

```text
   0    0    0    4
   0    0    2    4
   2  128    8    4
 256  128   64    2
```

Target: **LEFT**. Legal: UP, DOWN, LEFT.

Teacher action values (heuristic units): `{"UP": 1247179.88, "DOWN": 1529655.12, "LEFT": 1542100.62}`

## train-1027-337 (middle, teacher)

```text
  16   64  128  512
  16    4    0    0
   8    4    0    0
   4    0    0    2
```

Target: **LEFT**. Legal: UP, DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"UP": 1515633.0, "DOWN": 1309068.38, "LEFT": 1518241.62, "RIGHT": 1509902.0}`

## train-1068-2634 (late, teacher)

```text
   2    2    4   16
   4    4    8   64
   0    8   16  512
   2   16 1024 4096
```

Target: **UP**. Legal: UP, DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"UP": 1331531.0, "DOWN": 1198012.5, "LEFT": 1170228.5, "RIGHT": 1328460.12}`

## train-1076-231 (middle, priority)

```text
   0    0    0    0
   4    2    0    2
  16   32    8    4
  32  256  128   32
```

Target: **RIGHT**. Legal: UP, LEFT, RIGHT.

Teacher action values (heuristic units): `{"UP": 1206397.38, "LEFT": 1347423.62, "RIGHT": 1347631.38}`

## train-1078-10 (early, teacher)

```text
   2    4    4    4
   4    2    0    2
   2    0    0    0
   0    0    0    0
```

Target: **DOWN**. Legal: DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"DOWN": 1605586.88, "LEFT": 1601037.75, "RIGHT": 1604783.62}`

## train-1000-63 (early, random)

```text
   4    2    2    4
   0    8   16   16
   0   16   64    8
   0    2    0    4
```

Target: **DOWN**. Legal: DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"DOWN": 1482856.38, "LEFT": 1450956.62, "RIGHT": 1444309.62}`

## train-1066-45 (early, priority)

```text
   0    2    0    0
   4    0    0    0
   8    8    2    2
  32   32    4    4
```

Target: **DOWN**. Legal: UP, DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"UP": 1571742.0, "DOWN": 1590246.12, "LEFT": 1589004.75, "RIGHT": 1573616.62}`

## train-1073-433 (middle, teacher)

```text
 512  256  128    2
   4   16   32    8
   0    0    2    8
   0    2    0    2
```

Target: **RIGHT**. Legal: UP, DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"UP": 1463000.38, "DOWN": 1186121.0, "LEFT": 1460955.25, "RIGHT": 1463064.25}`

## train-1009-449 (middle, teacher)

```text
   0    2    2    8
   0    2    4   32
   2    4   16  256
   4    8  128  512
```

Target: **RIGHT**. Legal: UP, DOWN, LEFT, RIGHT.

Teacher action values (heuristic units): `{"UP": 1489698.5, "DOWN": 1491431.12, "LEFT": 1472552.0, "RIGHT": 1491539.38}`
