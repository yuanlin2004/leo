from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityProfile:
    name: str
    enabled_providers: frozenset[str]
    extra_system_prompt: str | None = None
    enable_file_tools: bool = True
    enable_shell_tools: bool = True
    enable_tmux_tools: bool = True
    enable_execution_tools: bool = False
    enable_skill_meta_tools: bool = True
    enable_mcp_meta_tools: bool = True

    def allows(self, provider_name: str, tags: frozenset[str]) -> bool:
        if provider_name not in self.enabled_providers:
            return False
        if "file" in tags and not self.enable_file_tools:
            return False
        if "shell" in tags and not self.enable_shell_tools:
            return False
        if "tmux" in tags and not self.enable_tmux_tools:
            return False
        if "execution" in tags and not self.enable_execution_tools:
            return False
        if "skills-meta" in tags and not self.enable_skill_meta_tools:
            return False
        if "mcp-meta" in tags and not self.enable_mcp_meta_tools:
            return False
        return True


BUILTIN_CAPABILITY_PROFILES: dict[str, CapabilityProfile] = {
    "generic": CapabilityProfile(
        name="generic",
        enabled_providers=frozenset({"local", "environment", "mcp", "skills"}),
    ),
    "benchmark-environment": CapabilityProfile(
        name="benchmark-environment",
        enabled_providers=frozenset({"local", "environment", "mcp"}),
        extra_system_prompt=(
            "\nYou are operating in a benchmark-oriented restricted environment. "
            "Use only the tools exposed by the selected profile or environment."
        ),
        enable_file_tools=False,
        enable_shell_tools=False,
        enable_tmux_tools=False,
        enable_execution_tools=True,
        enable_skill_meta_tools=False,
        enable_mcp_meta_tools=False,
    ),
}


def resolve_capability_profile(
    profile: CapabilityProfile | str | None,
) -> CapabilityProfile:
    if profile is None:
        return BUILTIN_CAPABILITY_PROFILES["generic"]
    if isinstance(profile, CapabilityProfile):
        return profile
    resolved = BUILTIN_CAPABILITY_PROFILES.get(profile)
    if resolved is None:
        available = ", ".join(sorted(BUILTIN_CAPABILITY_PROFILES))
        raise ValueError(f"Unknown capability profile: {profile}. Expected one of {available}.")
    return resolved
