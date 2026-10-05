import re

TOKEN = re.compile(r"'[^']*'|\S+")
OPS = {"=": lambda a, b: a == b, "<>": lambda a, b: a != b, "<": lambda a, b: a < b, ">": lambda a, b: a > b}


class Camera:
    def __init__(self, drives):
        self.drives = drives
        self.handles = {}
        self.next_handle = 0
        self.copies = []

    def path(self, name):
        return self.drives[name[0]].joinpath(*name[3:].split("\\"))

    def copy(self, src, dst):
        data = src.read_bytes()
        old = dst.read_bytes() if dst.exists() else b""
        dst.write_bytes(data + old[len(data):])

    def run(self, script):
        lines = [line.strip() for line in script.splitlines()]
        labels = {line[1:]: i for i, line in enumerate(lines) if line.startswith(":")}
        stack, endif_of = [], {}
        for i, line in enumerate(lines):
            head = line.split(" ", 1)[0]
            if head == "if":
                stack.append(i)
            elif head == "endif":
                endif_of[stack.pop()] = i
        assert not stack, "unbalanced if/endif"
        var = {"result": 0}

        def val(tok):
            if tok.startswith("'"):
                return tok[1:-1]
            if re.fullmatch(r"-?\d+", tok):
                return int(tok)
            return var[tok]

        pc = steps = 0
        while pc < len(lines):
            steps += 1
            assert steps < 200000, "runaway script"
            line, pc = lines[pc], pc + 1
            if not line or line[0] in ";:":
                continue
            t = TOKEN.findall(line)
            cmd = t[0]
            if len(t) == 3 and t[1] == "=":
                var[cmd] = val(t[2])
            elif cmd == "if":
                assert t[4] == "then" and len(t) == 5, line
                if not OPS[t[2]](val(t[1]), val(t[3])):
                    pc = endif_of[pc - 1] + 1
            elif cmd == "endif":
                pass
            elif cmd == "goto":
                pc = labels[t[1]]
            elif cmd == "exit":
                break
            elif cmd == "filesearch":
                var["result"] = int(self.path(val(t[1])).is_file())
            elif cmd == "filestat":
                p = self.path(val(t[1]))
                var["result"] = 0 if p.is_file() else -1
                if p.is_file():
                    var[t[2]] = p.stat().st_size
            elif cmd == "filecopy":
                src, dst = self.path(val(t[1])), self.path(val(t[2]))
                self.copies.append((val(t[1]), val(t[2])))
                self.copy(src, dst)
            elif cmd in ("filecreate", "fileopen"):
                p = self.path(val(t[2]))
                if cmd == "filecreate":
                    p.write_bytes(b"")
                elif not p.is_file():
                    var[t[1]] = -1
                    continue
                self.handles[self.next_handle] = [p, 0]
                var[t[1]] = self.next_handle
                self.next_handle += 1
            elif cmd == "filewrite":
                p = self.handles[val(t[1])][0]
                p.write_bytes(p.read_bytes() + str(val(t[2])).encode())
            elif cmd == "fileread":
                h = self.handles[val(t[1])]
                n = val(t[2])
                var[t[3]] = h[0].read_bytes()[h[1] : h[1] + n].decode()
                h[1] += n
            elif cmd == "fileclose":
                del self.handles[val(t[1])]
            elif cmd == "strcompare":
                a, b = val(t[1]), val(t[2])
                var["result"] = 0 if a == b else (1 if a > b else -1)
            else:
                raise AssertionError(f"unmodelled command: {line}")
        return var
