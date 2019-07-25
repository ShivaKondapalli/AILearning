import re

string = 'How are you and what is 9 plus 10 if not / then go and become madness + | {}'

pat = r'[^0-9]'

pattern = re.compile(pat)

print(re.search(pattern, string).re)
