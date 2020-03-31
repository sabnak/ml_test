from typing import Dict, List, Iterator, Callable, Union, Tuple, Optional
import regex


def findall(e: str, data: str, delimiter: str ="/", **kwargs) -> List:
	return regex.findall(compile(e, delimiter), data, **kwargs).groupdict()


def finditer(e: str, data: str, delimiter: str ="/", **kwargs) -> Iterator:
	return regex.finditer(compile(e, delimiter), data, **kwargs)


def finditer_to_dict(e: str, data: str, delimiter: str ="/", max_iterations: int =0, **kwargs) -> Callable:
	complied_regex = regex.compile(e, **kwargs)
	assert complied_regex.groupindex, "Only named group are supported"
	result = finditer(e, data, delimiter, **kwargs)
	i = 0

	def _iterator_to_dict():
		nonlocal i

		if max_iterations and i == max_iterations:
			raise StopIteration
		i += 1

		return next(result).groupdict()

	return _iterator_to_dict


def search(e: str, data: str, delimiter: str ="/", **kwargs) -> Union[Dict, None]:
	result = regex.search(compile(e, delimiter), data, **kwargs)
	if not result:
		return None

	return result.groupdict() or result.groups()


def sub_list(pairs: List[Tuple], data: str, delimiter: str ="/") -> str:
	for e, replacement, *kwargs in pairs:
		kwargs = kwargs[0] if kwargs else dict()
		data = regex.sub(compile(e, delimiter), replacement, data, **kwargs)

	return data


def compile(e, delimiter="/"):
	parsed_flags = 0
	flags = regex.search('^' + delimiter + '.+?' + delimiter + '([a-z]*)$', e)
	if flags:
		flags = flags.group(1)
		e = regex.sub(f'^{delimiter}|{delimiter}[a-z]*$', '', e)
		for flag in list(flags):
			parsed_flags |= getattr(regex, flag.upper())

	return regex.compile(e, parsed_flags)
