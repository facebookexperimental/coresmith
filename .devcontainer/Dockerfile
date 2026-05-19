# Codespaces-only extension of the published image. We can't modify the
# published :latest until publish-image starts succeeding again, so we
# layer the FHS sshd symlinks (PR #11's fix) on top at devcontainer
# build time. The base image is otherwise unchanged.
FROM ghcr.io/facebookexperimental/socmate:latest

RUN set -eux \
 && SSHD_BIN="$(command -v sshd)" \
 && SSH_KEYGEN_BIN="$(command -v ssh-keygen)" \
 && test -x "${SSHD_BIN}" \
 && test -x "${SSH_KEYGEN_BIN}" \
 && mkdir -p /usr/sbin /usr/bin /etc/ssh /var/run/sshd /run/sshd /var/empty \
 && ln -sf "${SSHD_BIN}" /usr/sbin/sshd \
 && ln -sf "${SSH_KEYGEN_BIN}" /usr/bin/ssh-keygen \
 && if ! getent passwd sshd >/dev/null 2>&1; then \
        if command -v useradd >/dev/null 2>&1; then \
            useradd -r -M -d /var/empty -s /usr/sbin/nologin sshd; \
        else \
            echo 'sshd:x:74:74:Privilege-separated SSH:/var/empty:/usr/sbin/nologin' >> /etc/passwd \
         && echo 'sshd:x:74:' >> /etc/group; \
        fi; \
    fi \
 && chmod 755 /var/empty \
 && ssh-keygen -A
